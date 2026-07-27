"""Tests for industrial IoT predictive-maintenance agents (#161).

Defines the expected API for MachineEnv, the spectral/health/RUL agents, the
Q-learning maintenance scheduler, and the MaintenanceOrchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def degrading_machine() -> tuple[np.ndarray, np.ndarray]:
    """Build a 2-sensor run whose vibration RMS grows toward failure, with health."""
    rng = np.random.default_rng(0)
    n = 60
    # Vibration amplitude ramps up over time (degradation); temperature drifts.
    t = np.linspace(0, 1, n)
    amp = 1.0 + 5.0 * t**2
    vibration = rng.normal(0, 1, n) * amp
    temperature = 40.0 + 20.0 * t + rng.normal(0, 0.5, n)
    sensors = np.column_stack([vibration, temperature])
    health = np.clip(1.0 - t**2, 0.0, 1.0)
    return sensors, health


@pytest.fixture
def healthy_machine() -> np.ndarray:
    """Build a stable 2-sensor run with no degradation."""
    rng = np.random.default_rng(1)
    n = 40
    sensors = np.column_stack([rng.normal(0, 1, n), 40.0 + rng.normal(0, 0.5, n)])
    return sensors


# ---------------------------------------------------------------------------
# MachineEnv
# ---------------------------------------------------------------------------


class TestMachineEnv:
    def test_creation(self, degrading_machine):
        from polars_ts.iiot_agents import MachineEnv

        sensors, health = degrading_machine
        env = MachineEnv(sensors, health=health)
        assert env.n_steps == 60
        assert env.n_sensors == 2

    def test_rejects_1d(self):
        from polars_ts.iiot_agents import MachineEnv

        with pytest.raises(ValueError, match="2D"):
            MachineEnv(np.zeros(10))

    def test_failure_step_inferred_from_health(self, degrading_machine):
        from polars_ts.iiot_agents import MachineEnv

        sensors, health = degrading_machine
        env = MachineEnv(sensors, health=health, failure_threshold=0.2)
        expected = int(np.nonzero(health <= 0.2)[0][0])
        assert env.failure_step == expected

    def test_invalid_action_raises(self, degrading_machine):
        from polars_ts.iiot_agents import MachineEnv

        sensors, health = degrading_machine
        env = MachineEnv(sensors, health=health)
        env.reset()
        with pytest.raises(ValueError, match="action"):
            env.step(7)

    def test_operate_penalized_after_failure(self, degrading_machine):
        from polars_ts.iiot_agents import MachineEnv
        from polars_ts.iiot_agents.env import OPERATE

        sensors, health = degrading_machine
        env = MachineEnv(sensors, health=health, failure_penalty=10.0)
        env.reset()
        rewards = []
        done = False
        while not done:
            _, r, done, _ = env.step(OPERATE)
            rewards.append(r)
        # Running to failure incurs a large negative reward at some point.
        assert min(rewards) <= -10.0 + 1e-9

    def test_timely_maintenance_beats_early(self, degrading_machine):
        from polars_ts.iiot_agents import MachineEnv
        from polars_ts.iiot_agents.env import MAINTAIN

        sensors, health = degrading_machine
        env = MachineEnv(sensors, health=health)
        fs = env.failure_step
        env.reset()
        rewards = {}
        idx = 0
        done = False
        while not done:
            _, r, done, info = env.step(MAINTAIN)
            rewards[idx] = r
            idx += 1
        # Maintaining just before failure is worth more than maintaining at t=0.
        assert rewards[fs - 1] > rewards[0]

    def test_episode_terminates(self, healthy_machine):
        from polars_ts.iiot_agents import MachineEnv
        from polars_ts.iiot_agents.env import OPERATE

        env = MachineEnv(healthy_machine)
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, done, _ = env.step(OPERATE)
            steps += 1
            assert steps <= len(healthy_machine)
        assert steps == len(healthy_machine)

    def test_nan_forward_filled(self):
        from polars_ts.iiot_agents import MachineEnv

        sensors = np.array([[1.0, 2.0], [np.nan, 2.0], [3.0, 2.0]])
        env = MachineEnv(sensors)
        assert not np.isnan(env.sensors).any()
        assert env.sensors[1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SpectralFeatureAgent
# ---------------------------------------------------------------------------


class TestSpectralFeatureAgent:
    def test_feature_length(self):
        from polars_ts.iiot_agents import SpectralFeatureAgent

        feats = SpectralFeatureAgent(n_bands=3).extract(np.random.default_rng(0).normal(0, 1, 64))
        assert feats.shape == (4,)  # rms + 3 band fractions

    def test_band_fractions_sum_to_one(self):
        from polars_ts.iiot_agents import SpectralFeatureAgent

        feats = SpectralFeatureAgent(n_bands=4).extract(np.random.default_rng(0).normal(0, 1, 128))
        assert feats[1:].sum() == pytest.approx(1.0, abs=1e-6)

    def test_rms_grows_with_amplitude(self):
        from polars_ts.iiot_agents import SpectralFeatureAgent

        agent = SpectralFeatureAgent()
        rng = np.random.default_rng(0)
        low = agent.extract(rng.normal(0, 1, 64))[0]
        high = agent.extract(rng.normal(0, 5, 64))[0]
        assert high > low

    def test_invalid_n_bands(self):
        from polars_ts.iiot_agents import SpectralFeatureAgent

        with pytest.raises(ValueError, match="n_bands"):
            SpectralFeatureAgent(n_bands=0)


# ---------------------------------------------------------------------------
# HealthIndexAgent
# ---------------------------------------------------------------------------


class TestHealthIndexAgent:
    def test_healthy_near_one(self, healthy_machine):
        from polars_ts.iiot_agents import HealthIndexAgent

        agent = HealthIndexAgent(warmup=5)
        agent.fit_baseline(healthy_machine)
        h = agent.score(healthy_machine[:5])
        assert h > 0.8

    def test_degradation_lowers_health(self, degrading_machine):
        from polars_ts.iiot_agents import HealthIndexAgent

        sensors, _ = degrading_machine
        agent = HealthIndexAgent(warmup=5)
        agent.fit_baseline(sensors)
        early = agent.score(sensors[:5])
        late = agent.score(sensors[-5:])
        assert late < early

    def test_bounded_zero_one(self, degrading_machine):
        from polars_ts.iiot_agents import HealthIndexAgent

        sensors, _ = degrading_machine
        agent = HealthIndexAgent(warmup=5)
        agent.fit_baseline(sensors)
        for i in range(len(sensors)):
            h = agent.score(sensors[max(0, i - 4) : i + 1])
            assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# RULEstimator
# ---------------------------------------------------------------------------


class TestRULEstimator:
    def test_infinite_when_stable(self):
        from polars_ts.iiot_agents import RULEstimator

        rul = RULEstimator().estimate([1.0, 1.0, 1.0, 1.0])
        assert rul == float("inf")

    def test_finite_when_declining(self):
        from polars_ts.iiot_agents import RULEstimator

        history = list(np.linspace(1.0, 0.5, 10))
        rul = RULEstimator(failure_threshold=0.2).estimate(history)
        assert 0.0 < rul < float("inf")

    def test_zero_when_below_threshold(self):
        from polars_ts.iiot_agents import RULEstimator

        rul = RULEstimator(failure_threshold=0.2).estimate([0.5, 0.3, 0.1])
        assert rul == 0.0

    def test_insufficient_history(self):
        from polars_ts.iiot_agents import RULEstimator

        assert RULEstimator(min_history=3).estimate([0.9, 0.8]) == float("inf")


# ---------------------------------------------------------------------------
# MaintenanceSchedulerAgent
# ---------------------------------------------------------------------------


class TestMaintenanceSchedulerAgent:
    def test_bucketing_bounds(self):
        from polars_ts.iiot_agents import MaintenanceSchedulerAgent

        agent = MaintenanceSchedulerAgent(n_states=10)
        assert agent.bucket(1.0) == 9
        assert agent.bucket(0.0) == 0
        assert 0 <= agent.bucket(0.5) <= 9

    def test_act_returns_valid_action(self):
        from polars_ts.iiot_agents import MaintenanceSchedulerAgent

        agent = MaintenanceSchedulerAgent()
        assert 0 <= agent.act(3) < agent.n_actions

    def test_q_update_moves_value(self):
        from polars_ts.iiot_agents import MaintenanceSchedulerAgent

        agent = MaintenanceSchedulerAgent()
        before = agent.q[2, 1]
        agent.update(2, 1, reward=5.0, next_state=2)
        assert agent.q[2, 1] > before

    def test_learns_to_maintain_when_degraded(self):
        from polars_ts.iiot_agents import MaintenanceSchedulerAgent
        from polars_ts.iiot_agents.env import MAINTAIN

        agent = MaintenanceSchedulerAgent(n_states=5)
        # Repeatedly reward maintaining in the most-degraded bucket.
        for _ in range(200):
            agent.update(0, MAINTAIN, reward=1.0, next_state=0)
        assert agent.act(0) == MAINTAIN


# ---------------------------------------------------------------------------
# MaintenanceOrchestrator
# ---------------------------------------------------------------------------


class TestMaintenanceOrchestrator:
    def test_run_returns_result(self, degrading_machine):
        from polars_ts.iiot_agents import MaintenanceOrchestrator, MaintenanceResult

        sensors, health = degrading_machine
        result = MaintenanceOrchestrator(n_episodes=30).run(sensors, health=health)
        assert isinstance(result, MaintenanceResult)

    def test_result_shapes(self, degrading_machine):
        from polars_ts.iiot_agents import MaintenanceOrchestrator

        sensors, health = degrading_machine
        result = MaintenanceOrchestrator(n_episodes=30).run(sensors, health=health)
        n = len(sensors)
        assert result.actions.shape == (n,)
        assert result.health_index.shape == (n,)
        assert result.rul.shape == (n,)
        assert len(result.history) == n

    def test_schedules_maintenance_before_failure(self, degrading_machine):
        from polars_ts.iiot_agents import MaintenanceOrchestrator
        from polars_ts.iiot_agents.env import MachineEnv

        sensors, health = degrading_machine
        result = MaintenanceOrchestrator(n_episodes=80, seed=0).run(sensors, health=health)
        fs = MachineEnv(sensors, health=health).failure_step
        # The learned policy should schedule maintenance, and do so before failure.
        assert result.first_maintenance_step >= 0
        assert result.first_maintenance_step <= fs

    def test_health_estimated_without_ground_truth(self, degrading_machine):
        from polars_ts.iiot_agents import MaintenanceOrchestrator

        sensors, _ = degrading_machine
        result = MaintenanceOrchestrator(n_episodes=20).run(sensors)
        # Estimated health should decline from start to end for a degrading machine.
        assert result.health_index[-5:].mean() < result.health_index[:5].mean()


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    NAMES = [
        "MachineEnv",
        "SpectralFeatureAgent",
        "HealthIndexAgent",
        "RULEstimator",
        "MaintenanceSchedulerAgent",
        "MaintenanceOrchestrator",
        "MaintenanceResult",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_module(self, name):
        import polars_ts.iiot_agents as mod

        assert getattr(mod, name) is not None
        assert name in mod.__all__

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_top_level(self, name):
        import polars_ts

        assert getattr(polars_ts, name) is not None
