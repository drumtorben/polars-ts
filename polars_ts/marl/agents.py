"""Specialized agents for multi-agent RL portfolio decisions."""

from __future__ import annotations

import numpy as np


class RiskAgent:
    """Assesses per-asset risk from recent return history.

    Uses rolling volatility (standard deviation of returns) as the risk metric.

    Parameters
    ----------
    window_size
        Number of recent periods to compute volatility over.

    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size

    def assess(self, returns: np.ndarray) -> np.ndarray:
        """Return per-asset risk scores (higher = riskier).

        Parameters
        ----------
        returns
            Array of shape ``(n_steps, n_assets)``.

        Returns
        -------
        np.ndarray
            Risk scores of shape ``(n_assets,)``.

        """
        recent = returns[-self.window_size :]
        return np.std(recent, axis=0)


class ReturnAgent:
    """Predicts expected per-asset returns from recent history.

    Uses exponentially weighted moving average of returns.

    Parameters
    ----------
    window_size
        Number of recent periods to consider.
    decay
        Exponential decay factor (higher = more weight on recent data).

    """

    def __init__(self, window_size: int = 20, decay: float = 0.94) -> None:
        self.window_size = window_size
        self.decay = decay

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Return expected per-asset returns.

        Parameters
        ----------
        returns
            Array of shape ``(n_steps, n_assets)``.

        Returns
        -------
        np.ndarray
            Expected returns of shape ``(n_assets,)``.

        """
        recent = returns[-self.window_size :]
        n = len(recent)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return weights @ recent


class AllocationAgent:
    """Produces portfolio weights from risk and return estimates.

    Uses a risk-adjusted return score (return / risk) to allocate weights
    proportionally.

    Parameters
    ----------
    risk_aversion
        Higher values penalize risk more.

    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        self.risk_aversion = risk_aversion

    def allocate(
        self,
        risk_scores: np.ndarray,
        expected_returns: np.ndarray,
        n_assets: int,  # noqa: ARG002
    ) -> np.ndarray:
        """Return normalized portfolio weights.

        Parameters
        ----------
        risk_scores
            Per-asset risk scores from ``RiskAgent``.
        expected_returns
            Per-asset expected returns from ``ReturnAgent``.
        n_assets
            Number of assets (must match array lengths).

        Returns
        -------
        np.ndarray
            Portfolio weights summing to 1, shape ``(n_assets,)``.

        """
        safe_risk = np.maximum(risk_scores, 1e-10)
        scores = expected_returns / (safe_risk * self.risk_aversion)
        # Shift to positive then normalize
        shifted = scores - scores.min() + 1e-10
        weights = shifted / shifted.sum()
        return weights
