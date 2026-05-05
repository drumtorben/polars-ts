"""MARLOrchestrator: chains risk, return, and allocation agents over a PortfolioEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from polars_ts.marl.agents import AllocationAgent, ReturnAgent, RiskAgent
from polars_ts.marl.env import PortfolioEnv


@dataclass
class MARLResult:
    """Output of a MARLOrchestrator run."""

    weights_history: np.ndarray
    portfolio_returns: np.ndarray
    sharpe_ratio: float
    total_return: float
    history: list[dict[str, Any]] = field(default_factory=list)


class MARLOrchestrator:
    """Orchestrates specialized agents over a portfolio environment.

    At each step:
    1. RiskAgent assesses per-asset volatility.
    2. ReturnAgent predicts expected returns.
    3. AllocationAgent combines risk/return into portfolio weights.
    4. PortfolioEnv evaluates the resulting allocation.

    Parameters
    ----------
    window_size
        Observation window for the environment and agents.
    risk_aversion
        Passed to ``AllocationAgent``.
    transaction_cost
        Proportional cost for weight changes.

    """

    def __init__(
        self,
        window_size: int = 10,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.0,
    ) -> None:
        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

    def run(self, returns: np.ndarray) -> MARLResult:
        """Run the multi-agent loop over a returns matrix.

        Parameters
        ----------
        returns
            Array of shape ``(n_steps, n_assets)``.

        Returns
        -------
        MARLResult

        """
        env = PortfolioEnv(returns, window_size=self.window_size, transaction_cost=self.transaction_cost)
        return self._run_loop(env, returns)

    def run_from_dataframe(
        self,
        df: pl.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> MARLResult:
        """Run from a polars panel DataFrame of prices."""
        env = PortfolioEnv.from_dataframe(
            df,
            window_size=self.window_size,
            transaction_cost=self.transaction_cost,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        return self._run_loop(env, env.returns)

    def _run_loop(self, env: PortfolioEnv, returns: np.ndarray) -> MARLResult:
        risk_agent = RiskAgent(window_size=self.window_size)
        return_agent = ReturnAgent(window_size=self.window_size)
        alloc_agent = AllocationAgent(risk_aversion=self.risk_aversion)

        history: list[dict[str, Any]] = []
        weights_list: list[np.ndarray] = []
        returns_list: list[float] = []

        obs = env.reset()
        done = False

        while not done:
            step_idx = env._step
            # Agents observe the full history up to current step
            hist_end = self.window_size + step_idx
            hist_returns = returns[:hist_end]

            risk_scores = risk_agent.assess(hist_returns)
            history.append({"agent": "risk", "message": f"Risk scores: {risk_scores.tolist()}"})

            expected_rets = return_agent.predict(hist_returns)
            history.append({"agent": "return", "message": f"Expected returns: {expected_rets.tolist()}"})

            weights = alloc_agent.allocate(risk_scores, expected_rets, env.n_assets)
            history.append({"agent": "allocation", "message": f"Weights: {weights.tolist()}"})

            obs, reward, done, info = env.step(weights)
            weights_list.append(weights)
            returns_list.append(info["portfolio_return"])

        weights_history = np.array(weights_list)
        portfolio_returns = np.array(returns_list)

        # Compute metrics
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std() + 1e-10
        sharpe_ratio = float(mean_ret / std_ret * np.sqrt(252))  # annualized
        total_return = float(np.prod(1 + portfolio_returns) - 1)

        return MARLResult(
            weights_history=weights_history,
            portfolio_returns=portfolio_returns,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            history=history,
        )
