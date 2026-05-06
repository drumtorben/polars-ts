"""Tests for multi-agent RL framework (#158).

TDD: tests define the expected API for PortfolioEnv, specialized agents,
and the MARLOrchestrator before implementation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def price_panel() -> pl.DataFrame:
    """Multi-asset daily price panel (3 assets, 100 days)."""
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    n = 100
    base = date(2024, 1, 1)
    rows: list[dict] = []
    for asset in ["AAPL", "GOOG", "MSFT"]:
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        for t in range(n):
            rows.append({"unique_id": asset, "ds": base + timedelta(days=t), "y": float(prices[t])})
    return pl.DataFrame(rows)


@pytest.fixture
def returns_matrix() -> np.ndarray:
    """Create a returns matrix of shape (n_steps, n_assets)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, (100, 3))


# ---------------------------------------------------------------------------
# PortfolioEnv tests
# ---------------------------------------------------------------------------


class TestPortfolioEnv:
    def test_creation(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10)
        assert env._max_steps > 0

    def test_reset_returns_observation(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        # obs shape: (window_size, n_assets)
        assert obs.shape == (10, 3)

    def test_step_returns_tuple(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10)
        env.reset()
        # action is portfolio weights: (n_assets,)
        action = np.array([0.4, 0.3, 0.3])
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "portfolio_return" in info
        assert "weights" in info

    def test_weights_normalized(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10)
        env.reset()
        action = np.array([2.0, 3.0, 5.0])  # not normalized
        _, _, _, info = env.step(action)
        assert abs(sum(info["weights"]) - 1.0) < 1e-10

    def test_episode_terminates(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10)
        env.reset()
        action = np.array([1 / 3, 1 / 3, 1 / 3])
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps == len(returns_matrix) - 10

    def test_custom_transaction_cost(self, returns_matrix):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv(returns_matrix, window_size=10, transaction_cost=0.001)
        env.reset()
        action = np.array([0.5, 0.3, 0.2])
        _, r1, _, _ = env.step(action)
        # Change allocation → incur cost
        action2 = np.array([0.2, 0.5, 0.3])
        _, r2, _, _ = env.step(action2)
        assert isinstance(r2, float)

    def test_from_dataframe(self, price_panel):
        from polars_ts.marl import PortfolioEnv

        env = PortfolioEnv.from_dataframe(price_panel, window_size=10)
        obs = env.reset()
        assert obs.shape[1] == 3  # 3 assets


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestRiskAgent:
    def test_assess_returns_risk_scores(self, returns_matrix):
        from polars_ts.marl import RiskAgent

        agent = RiskAgent(window_size=20)
        scores = agent.assess(returns_matrix[:30])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == returns_matrix.shape[1]
        assert np.all(np.isfinite(scores))

    def test_risk_higher_for_volatile_asset(self):
        from polars_ts.marl import RiskAgent

        rng = np.random.default_rng(42)
        # Asset 0: low vol, Asset 1: high vol
        returns = np.column_stack([rng.normal(0, 0.01, 50), rng.normal(0, 0.1, 50)])
        agent = RiskAgent(window_size=20)
        scores = agent.assess(returns)
        assert scores[1] > scores[0]  # higher risk for volatile asset


class TestReturnAgent:
    def test_predict_returns_expected_returns(self, returns_matrix):
        from polars_ts.marl import ReturnAgent

        agent = ReturnAgent(window_size=20)
        expected = agent.predict(returns_matrix[:30])
        assert isinstance(expected, np.ndarray)
        assert len(expected) == returns_matrix.shape[1]
        assert np.all(np.isfinite(expected))


class TestAllocationAgent:
    def test_allocate_returns_weights(self, returns_matrix):
        from polars_ts.marl import AllocationAgent

        n_assets = returns_matrix.shape[1]
        risk_scores = np.array([0.5, 0.3, 0.2])
        expected_returns = np.array([0.01, 0.02, 0.015])

        agent = AllocationAgent()
        weights = agent.allocate(risk_scores, expected_returns, n_assets)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == n_assets
        assert abs(weights.sum() - 1.0) < 1e-10
        assert np.all(weights >= 0)

    def test_favors_high_return_low_risk(self):
        from polars_ts.marl import AllocationAgent

        agent = AllocationAgent()
        risk = np.array([0.1, 0.5, 0.3])
        ret = np.array([0.05, 0.01, 0.03])
        weights = agent.allocate(risk, ret, 3)
        # Asset 0 has best return/risk ratio → should get highest weight
        assert weights[0] > weights[1]


# ---------------------------------------------------------------------------
# MARLOrchestrator tests
# ---------------------------------------------------------------------------


class TestMARLOrchestrator:
    def test_run_returns_result(self, returns_matrix):
        from polars_ts.marl import MARLOrchestrator, MARLResult

        orch = MARLOrchestrator(window_size=10)
        result = orch.run(returns_matrix)
        assert isinstance(result, MARLResult)

    def test_result_has_weights_history(self, returns_matrix):
        from polars_ts.marl import MARLOrchestrator

        orch = MARLOrchestrator(window_size=10)
        result = orch.run(returns_matrix)
        assert isinstance(result.weights_history, np.ndarray)
        n_steps = len(returns_matrix) - 10
        assert result.weights_history.shape == (n_steps, returns_matrix.shape[1])

    def test_result_has_portfolio_returns(self, returns_matrix):
        from polars_ts.marl import MARLOrchestrator

        orch = MARLOrchestrator(window_size=10)
        result = orch.run(returns_matrix)
        assert isinstance(result.portfolio_returns, np.ndarray)
        assert len(result.portfolio_returns) == len(returns_matrix) - 10

    def test_result_has_metrics(self, returns_matrix):
        from polars_ts.marl import MARLOrchestrator

        orch = MARLOrchestrator(window_size=10)
        result = orch.run(returns_matrix)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.total_return, float)
        assert np.isfinite(result.sharpe_ratio)

    def test_from_dataframe(self, price_panel):
        from polars_ts.marl import MARLOrchestrator

        orch = MARLOrchestrator(window_size=10)
        result = orch.run_from_dataframe(price_panel)
        assert result.weights_history.shape[1] == 3

    def test_history_logged(self, returns_matrix):
        from polars_ts.marl import MARLOrchestrator

        orch = MARLOrchestrator(window_size=10)
        result = orch.run(returns_matrix)
        assert len(result.history) >= 3  # risk, return, allocation agents logged


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_importable_from_marl(self):
        from polars_ts.marl import MARLOrchestrator, PortfolioEnv

        assert PortfolioEnv is not None
        assert MARLOrchestrator is not None

    def test_importable_from_top_level(self):
        from polars_ts import MARLOrchestrator, PortfolioEnv

        assert PortfolioEnv is not None
        assert MARLOrchestrator is not None
