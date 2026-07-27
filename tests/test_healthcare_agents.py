"""Tests for multi-agent clinical decision support (#160).

Defines the expected API for ClinicalEnv, the clinical agents, federated
averaging, and the ClinicalOrchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stable_row() -> list[float]:
    """Return a physiologically normal vital-sign row."""
    # heart_rate, systolic_bp, respiratory_rate, temperature, spo2
    return [75.0, 120.0, 16.0, 37.0, 98.0]


def _septic_row() -> list[float]:
    """Return a deranged row consistent with sepsis / deterioration."""
    return [125.0, 92.0, 26.0, 39.2, 91.0]


@pytest.fixture
def stable_vitals() -> np.ndarray:
    """Build a stable trajectory with no deterioration."""
    return np.array([_stable_row() for _ in range(30)], dtype=np.float64)


@pytest.fixture
def deteriorating_vitals() -> tuple[np.ndarray, np.ndarray]:
    """Build a trajectory that deteriorates in its second half, with labels."""
    rows = [_stable_row() for _ in range(15)] + [_septic_row() for _ in range(15)]
    labels = np.array([False] * 15 + [True] * 15, dtype=bool)
    return np.array(rows, dtype=np.float64), labels


# ---------------------------------------------------------------------------
# ClinicalEnv
# ---------------------------------------------------------------------------


class TestClinicalEnv:
    def test_creation(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        env = ClinicalEnv(stable_vitals)
        assert env.n_steps == 30
        assert env.n_channels == 5

    def test_reset_returns_first_row(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        env = ClinicalEnv(stable_vitals)
        obs = env.reset()
        assert obs.shape == (5,)
        np.testing.assert_allclose(obs, _stable_row())

    def test_rejects_1d_input(self):
        from polars_ts.healthcare_agents import ClinicalEnv

        with pytest.raises(ValueError, match="2D"):
            ClinicalEnv(np.zeros(10))

    def test_step_returns_tuple(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        env = ClinicalEnv(stable_vitals)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "elapsed" in info and "tier" in info

    def test_invalid_tier_raises(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        env = ClinicalEnv(stable_vitals)
        env.reset()
        with pytest.raises(ValueError, match="tier"):
            env.step(99)

    def test_episode_terminates(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        env = ClinicalEnv(stable_vitals)
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
            assert steps <= 30
        assert steps == 30

    def test_irregular_times_reported(self):
        from polars_ts.healthcare_agents import ClinicalEnv

        vitals = np.array([_stable_row() for _ in range(3)], dtype=np.float64)
        times = np.array([0.0, 2.5, 10.0])
        env = ClinicalEnv(vitals, times=times)
        env.reset()
        _, _, _, info0 = env.step(0)
        assert info0["elapsed"] == 0.0
        _, _, _, info1 = env.step(0)
        assert info1["elapsed"] == pytest.approx(2.5)

    def test_nan_forward_filled(self):
        from polars_ts.healthcare_agents import ClinicalEnv

        vitals = np.array([_stable_row(), _stable_row(), _stable_row()], dtype=np.float64)
        vitals[1, 0] = np.nan  # missing heart rate mid-stream
        env = ClinicalEnv(vitals)
        assert not np.isnan(env.vitals).any()
        assert env.vitals[1, 0] == pytest.approx(vitals[0, 0])

    def test_reward_rewards_correct_escalation(self, deteriorating_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        vitals, labels = deteriorating_vitals
        env = ClinicalEnv(vitals, labels=labels)
        env.reset()
        # Advance to a deteriorating step; escalating should beat ignoring.
        for _ in range(20):
            env.step(0)
        r_escalate = env._reward(20, tier=3)
        r_ignore = env._reward(20, tier=0)
        assert r_escalate > r_ignore

    def test_reward_penalizes_alarm_fatigue(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalEnv

        labels = np.zeros(len(stable_vitals), dtype=bool)
        env = ClinicalEnv(stable_vitals, labels=labels)
        assert env._reward(0, tier=0) > env._reward(0, tier=3)


# ---------------------------------------------------------------------------
# SepsisWarningAgent
# ---------------------------------------------------------------------------


class TestSepsisWarningAgent:
    def test_flags_septic_row(self):
        from polars_ts.healthcare_agents import SepsisWarningAgent

        agent = SepsisWarningAgent()
        risk, flag = agent.score(np.array(_septic_row()))
        assert flag is True
        assert risk >= 2.0

    def test_ignores_stable_row(self):
        from polars_ts.healthcare_agents import SepsisWarningAgent

        agent = SepsisWarningAgent()
        _, flag = agent.score(np.array(_stable_row()))
        assert flag is False


# ---------------------------------------------------------------------------
# VitalMonitorAgent
# ---------------------------------------------------------------------------


class TestVitalMonitorAgent:
    def test_no_derangement_when_stable(self):
        from polars_ts.healthcare_agents import VitalMonitorAgent

        n, any_flag = VitalMonitorAgent().score(np.array(_stable_row()))
        assert n == 0.0
        assert any_flag is False

    def test_counts_deranged_channels(self):
        from polars_ts.healthcare_agents import VitalMonitorAgent

        n, any_flag = VitalMonitorAgent().score(np.array(_septic_row()))
        assert n >= 3.0
        assert any_flag is True


# ---------------------------------------------------------------------------
# EscalationAgent
# ---------------------------------------------------------------------------


class TestEscalationAgent:
    def test_stable_maps_to_tier_zero(self):
        from polars_ts.healthcare_agents import EscalationAgent

        tier = EscalationAgent().decide(np.array(_stable_row()), sepsis_risk=False, n_deranged=0.0)
        assert tier == 0

    def test_septic_maps_to_high_tier(self):
        from polars_ts.healthcare_agents import EscalationAgent

        tier = EscalationAgent().decide(np.array(_septic_row()), sepsis_risk=True, n_deranged=4.0)
        assert tier == 3

    def test_sepsis_forces_review(self):
        from polars_ts.healthcare_agents import EscalationAgent

        agent = EscalationAgent()
        # Otherwise-stable vitals but sepsis risk flagged -> at least urgent review.
        tier = agent.decide(np.array(_stable_row()), sepsis_risk=True, n_deranged=0.0)
        assert tier >= 2


# ---------------------------------------------------------------------------
# TreatmentAgent
# ---------------------------------------------------------------------------


class TestTreatmentAgent:
    def test_recommend_returns_valid_action(self):
        from polars_ts.healthcare_agents import TreatmentAgent

        agent = TreatmentAgent()
        a = agent.recommend(tier=2)
        assert 0 <= a < len(agent.actions)

    def test_learns_from_reward(self):
        from polars_ts.healthcare_agents import TreatmentAgent

        agent = TreatmentAgent()
        # Reinforce action 3 for tier 3 repeatedly; it should become preferred.
        for _ in range(20):
            agent.update(tier=3, action=3, reward=1.0)
        assert agent.recommend(tier=3) == 3

    def test_deterministic_without_exploration(self):
        from polars_ts.healthcare_agents import TreatmentAgent

        agent = TreatmentAgent(seed=1)
        assert agent.recommend(tier=1) == agent.recommend(tier=1)


# ---------------------------------------------------------------------------
# federated_average
# ---------------------------------------------------------------------------


class TestFederatedAverage:
    def test_equal_weight_mean(self):
        from polars_ts.healthcare_agents import federated_average

        a = np.array([[0.0, 2.0], [4.0, 6.0]])
        b = np.array([[2.0, 4.0], [6.0, 8.0]])
        out = federated_average([a, b])
        np.testing.assert_allclose(out, [[1.0, 3.0], [5.0, 7.0]])

    def test_weighted_mean(self):
        from polars_ts.healthcare_agents import federated_average

        a = np.zeros((2, 2))
        b = np.ones((2, 2))
        out = federated_average([a, b], weights=[1.0, 3.0])
        np.testing.assert_allclose(out, np.full((2, 2), 0.75))

    def test_shape_mismatch_raises(self):
        from polars_ts.healthcare_agents import federated_average

        with pytest.raises(ValueError, match="same shape"):
            federated_average([np.zeros((2, 2)), np.zeros((3, 3))])

    def test_empty_raises(self):
        from polars_ts.healthcare_agents import federated_average

        with pytest.raises(ValueError, match="at least one"):
            federated_average([])


# ---------------------------------------------------------------------------
# ClinicalOrchestrator
# ---------------------------------------------------------------------------


class TestClinicalOrchestrator:
    def test_run_returns_result(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator, ClinicalResult

        result = ClinicalOrchestrator().run(stable_vitals)
        assert isinstance(result, ClinicalResult)

    def test_result_shapes(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator

        result = ClinicalOrchestrator().run(stable_vitals)
        assert result.escalation_tiers.shape == (30,)
        assert result.sepsis_flags.shape == (30,)
        assert len(result.treatments) == 30
        assert len(result.history) == 30

    def test_stable_patient_stays_calm(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator

        result = ClinicalOrchestrator().run(stable_vitals)
        assert result.escalation_tiers.max() == 0
        assert not result.sepsis_flags.any()

    def test_detects_deterioration(self, deteriorating_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator

        vitals, labels = deteriorating_vitals
        result = ClinicalOrchestrator().run(vitals, labels=labels)
        # Escalation must rise for the deteriorating second half.
        assert result.escalation_tiers[:15].max() < result.escalation_tiers[15:].max()
        assert result.sepsis_flags[15:].any()

    def test_recommends_icu_for_deterioration(self, deteriorating_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator

        vitals, labels = deteriorating_vitals
        result = ClinicalOrchestrator().run(vitals, labels=labels)
        assert "transfer_icu" in result.treatments

    def test_handles_irregular_times(self, stable_vitals):
        from polars_ts.healthcare_agents import ClinicalOrchestrator

        times = np.cumsum(np.linspace(0.5, 2.0, len(stable_vitals)))
        result = ClinicalOrchestrator().run(stable_vitals, times=times)
        assert len(result.history) == len(stable_vitals)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    NAMES = [
        "ClinicalEnv",
        "ClinicalOrchestrator",
        "ClinicalResult",
        "SepsisWarningAgent",
        "VitalMonitorAgent",
        "EscalationAgent",
        "TreatmentAgent",
        "federated_average",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_module(self, name):
        import polars_ts.healthcare_agents as mod

        assert getattr(mod, name) is not None
        assert name in mod.__all__

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_top_level(self, name):
        import polars_ts

        assert getattr(polars_ts, name) is not None
