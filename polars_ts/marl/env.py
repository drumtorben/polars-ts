"""Portfolio environment for multi-agent RL decision making."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class PortfolioEnv:
    """Gymnasium-like environment for portfolio allocation.

    At each step, the agent observes recent returns for all assets and
    produces portfolio weights. The reward is the portfolio return minus
    transaction costs.

    Parameters
    ----------
    returns
        Array of shape ``(n_steps, n_assets)`` with per-period returns.
    window_size
        Number of recent return periods provided as the observation.
    transaction_cost
        Proportional cost applied to weight changes between steps.

    """

    def __init__(
        self,
        returns: np.ndarray,
        window_size: int = 10,
        transaction_cost: float = 0.0,
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float64)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.n_assets = self.returns.shape[1]
        self._step = 0
        self._max_steps = len(self.returns) - window_size
        self._prev_weights = np.ones(self.n_assets) / self.n_assets

        if self._max_steps <= 0:
            raise ValueError("returns must have more rows than window_size")

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        window_size: int = 10,
        transaction_cost: float = 0.0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> PortfolioEnv:
        """Create environment from a polars panel DataFrame of prices.

        Converts prices to log returns and pivots to wide format.
        """
        sorted_df = df.sort(id_col, time_col)
        assets = sorted_df[id_col].unique(maintain_order=True).to_list()
        returns_list = []
        for asset in assets:
            prices = sorted_df.filter(pl.col(id_col) == asset)[target_col].to_numpy().astype(np.float64)
            if np.any(prices <= 0):
                raise ValueError(f"Prices for {asset} contain zero or negative values — cannot compute log returns")
            rets = np.diff(np.log(prices))
            returns_list.append(rets)

        min_len = min(len(r) for r in returns_list)
        returns = np.column_stack([r[:min_len] for r in returns_list])
        return cls(returns, window_size=window_size, transaction_cost=transaction_cost)

    def reset(self) -> np.ndarray:
        """Reset and return initial observation of shape ``(window_size, n_assets)``."""
        self._step = 0
        self._prev_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Take one step with portfolio weights.

        Parameters
        ----------
        action
            Raw portfolio weights (will be normalized to sum to 1).

        Returns
        -------
        tuple
            ``(observation, reward, done, info)``

        """
        weights = np.abs(action)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        idx = self.window_size + self._step
        period_returns = self.returns[idx]
        portfolio_return = float(np.dot(weights, period_returns))

        # Transaction cost from weight changes
        turnover = float(np.sum(np.abs(weights - self._prev_weights)))
        cost = self.transaction_cost * turnover
        reward = portfolio_return - cost

        self._prev_weights = weights.copy()
        self._step += 1
        done = self._step >= self._max_steps

        obs = self._get_obs() if not done else np.zeros((self.window_size, self.n_assets))
        info = {"portfolio_return": portfolio_return, "weights": weights.tolist(), "turnover": turnover}

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build observation: recent returns window of shape ``(window_size, n_assets)``."""
        start = self._step
        end = start + self.window_size
        return self.returns[start:end].copy()
