"""Tests for RL-based autonomous anomaly detection agents (#159).

TDD: tests define the expected API for AnomalyEnv, detection agents,
and the AnomalyOrchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_series() -> np.ndarray:
    """Stationary series with known anomalies injected."""
    rng = np.random.default_rng(42)
    n = 200
    values = rng.normal(0, 1, n)
    # Inject anomalies at known positions
    values[50] = 10.0
    values[100] = -10.0
    values[150] = 12.0
    return values


@pytest.fixture
def multivariate_series() -> np.ndarray:
    """Multi-channel series (n_steps, n_channels)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, (200, 3))
    values[50, 0] = 10.0
    values[100, 1] = -10.0
    return values


# ---------------------------------------------------------------------------
# AnomalyEnv tests
# ---------------------------------------------------------------------------


class TestAnomalyEnv:
    def test_creation(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyEnv

        env = AnomalyEnv(normal_series, window_size=20)
        assert env._max_steps > 0

    def test_reset_returns_observation(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyEnv

        env = AnomalyEnv(normal_series, window_size=20)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert len(obs) == 20

    def test_step_returns_tuple(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyEnv

        env = AnomalyEnv(normal_series, window_size=20)
        env.reset()
        obs, reward, done, info = env.step(False)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "value" in info
        assert "threshold" in info

    def test_episode_terminates(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyEnv

        env = AnomalyEnv(normal_series, window_size=20)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(False)
            steps += 1
        assert steps == len(normal_series) - 20

    def test_reward_for_correct_detection(self, normal_series):
        """Flagging a true anomaly should give positive reward."""
        from polars_ts.anomaly_agents import AnomalyEnv

        labels = np.zeros(len(normal_series), dtype=bool)
        labels[50] = True
        labels[100] = True
        labels[150] = True
        env = AnomalyEnv(normal_series, window_size=20, labels=labels)
        env.reset()
        # Step to position 50 (step index 30, since window=20)
        for _ in range(30):
            env.step(False)
        # Now at index 50 — flag it
        _, reward, _, _ = env.step(True)
        assert reward > 0

    def test_reward_for_false_alarm(self, normal_series):
        """Flagging a normal point should give negative reward."""
        from polars_ts.anomaly_agents import AnomalyEnv

        labels = np.zeros(len(normal_series), dtype=bool)
        env = AnomalyEnv(normal_series, window_size=20, labels=labels)
        env.reset()
        _, reward, _, _ = env.step(True)  # flag normal point
        assert reward < 0


# ---------------------------------------------------------------------------
# Detection agent tests
# ---------------------------------------------------------------------------


class TestZScoreAgent:
    def test_detect_returns_scores(self, normal_series):
        from polars_ts.anomaly_agents import ZScoreAgent

        agent = ZScoreAgent(threshold=3.0)
        window = normal_series[:30]
        score, flag = agent.detect(window)
        assert isinstance(score, float)
        assert isinstance(flag, bool)

    def test_detects_outlier(self):
        from polars_ts.anomaly_agents import ZScoreAgent

        agent = ZScoreAgent(threshold=3.0)
        window = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, 20.0])
        score, flag = agent.detect(window)
        assert flag is True
        assert score > 3.0

    def test_does_not_flag_normal(self):
        from polars_ts.anomaly_agents import ZScoreAgent

        agent = ZScoreAgent(threshold=3.0)
        window = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, 0.05])
        _, flag = agent.detect(window)
        assert flag is False


class TestRollingStdAgent:
    def test_detect_returns_scores(self, normal_series):
        from polars_ts.anomaly_agents import RollingStdAgent

        agent = RollingStdAgent(threshold=3.0)
        window = normal_series[:30]
        score, flag = agent.detect(window)
        assert isinstance(score, float)
        assert isinstance(flag, bool)
        assert np.isfinite(score)


class TestMADAgent:
    def test_detect_returns_scores(self, normal_series):
        from polars_ts.anomaly_agents import MADAgent

        agent = MADAgent(threshold=3.5)
        window = normal_series[:30]
        score, flag = agent.detect(window)
        assert isinstance(score, float)
        assert isinstance(flag, bool)

    def test_robust_to_outliers(self):
        """MAD should be robust — not influenced by the outlier itself."""
        from polars_ts.anomaly_agents import MADAgent

        agent = MADAgent(threshold=3.0)
        window = np.concatenate([np.zeros(19), [100.0]])
        score, flag = agent.detect(window)
        assert flag is True


# ---------------------------------------------------------------------------
# ConsensusAgent tests
# ---------------------------------------------------------------------------


class TestConsensusAgent:
    def test_majority_vote(self):
        from polars_ts.anomaly_agents import ConsensusAgent

        agent = ConsensusAgent(method="majority")
        flags = [True, True, False]
        scores = [5.0, 4.0, 1.0]
        result = agent.decide(flags, scores)
        assert result is True  # 2/3 say anomaly

    def test_majority_vote_normal(self):
        from polars_ts.anomaly_agents import ConsensusAgent

        agent = ConsensusAgent(method="majority")
        flags = [False, True, False]
        scores = [1.0, 4.0, 1.0]
        result = agent.decide(flags, scores)
        assert result is False

    def test_any_vote(self):
        from polars_ts.anomaly_agents import ConsensusAgent

        agent = ConsensusAgent(method="any")
        flags = [False, False, True]
        scores = [1.0, 1.0, 5.0]
        result = agent.decide(flags, scores)
        assert result is True

    def test_weighted_vote(self):
        from polars_ts.anomaly_agents import ConsensusAgent

        agent = ConsensusAgent(method="weighted", weights=[0.5, 0.3, 0.2])
        flags = [True, False, False]
        scores = [5.0, 1.0, 1.0]
        # 0.5 > 0.3+0.2 threshold → should flag
        result = agent.decide(flags, scores)
        assert result is True


# ---------------------------------------------------------------------------
# AnomalyOrchestrator tests
# ---------------------------------------------------------------------------


class TestAnomalyOrchestrator:
    def test_run_returns_result(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyOrchestrator, AnomalyResult

        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(normal_series)
        assert isinstance(result, AnomalyResult)

    def test_result_has_detections(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyOrchestrator

        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(normal_series)
        assert isinstance(result.detections, np.ndarray)
        assert result.detections.dtype == bool
        assert len(result.detections) == len(normal_series) - 20

    def test_result_has_scores(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyOrchestrator

        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(normal_series)
        assert isinstance(result.agent_scores, dict)
        assert len(result.agent_scores) >= 2

    def test_detects_injected_anomalies(self, normal_series):
        """Should detect at least some of the injected anomalies."""
        from polars_ts.anomaly_agents import AnomalyOrchestrator

        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(normal_series)
        # Anomalies at indices 50, 100, 150 → step indices 30, 80, 130
        assert result.detections.sum() >= 2  # detect at least 2 of 3

    def test_result_has_history(self, normal_series):
        from polars_ts.anomaly_agents import AnomalyOrchestrator

        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(normal_series)
        assert len(result.history) > 0

    def test_few_false_positives(self):
        """Clean series should have very few false positives."""
        from polars_ts.anomaly_agents import AnomalyOrchestrator

        rng = np.random.default_rng(42)
        clean = rng.normal(0, 1, 200)
        orch = AnomalyOrchestrator(window_size=20)
        result = orch.run(clean)
        # Should flag fewer than 5% as anomalies
        rate = result.detections.sum() / len(result.detections)
        assert rate < 0.05


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_importable_from_module(self):
        from polars_ts.anomaly_agents import AnomalyEnv, AnomalyOrchestrator

        assert AnomalyEnv is not None
        assert AnomalyOrchestrator is not None

    def test_importable_from_top_level(self):
        from polars_ts import AnomalyEnv, AnomalyOrchestrator

        assert AnomalyEnv is not None
        assert AnomalyOrchestrator is not None
