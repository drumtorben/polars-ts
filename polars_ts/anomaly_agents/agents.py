"""Specialized anomaly detection agents and consensus voting."""

from __future__ import annotations

import numpy as np


class ZScoreAgent:
    """Detect anomalies via z-score of the last value in a window.

    Parameters
    ----------
    threshold
        Z-score threshold above which a point is flagged.

    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def detect(self, window: np.ndarray) -> tuple[float, bool]:
        """Score the last value in the window.

        Returns
        -------
        tuple
            ``(z_score, is_anomaly)``

        """
        context = window[:-1]
        value = window[-1]
        mean = context.mean()
        std = context.std() + 1e-10
        z = abs(value - mean) / std
        return float(z), bool(z > self.threshold)


class RollingStdAgent:
    """Detect anomalies by comparing the last value to a rolling standard deviation.

    Parameters
    ----------
    threshold
        Multiplier of rolling std above which a point is flagged.

    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def detect(self, window: np.ndarray) -> tuple[float, bool]:
        """Score the last value in the window.

        Returns
        -------
        tuple
            ``(deviation_score, is_anomaly)``

        """
        context = window[:-1]
        value = window[-1]
        median = float(np.median(context))
        std = float(context.std()) + 1e-10
        score = abs(value - median) / std
        return float(score), bool(score > self.threshold)


class MADAgent:
    """Detect anomalies via Median Absolute Deviation (robust to outliers).

    Parameters
    ----------
    threshold
        Modified z-score threshold.

    """

    def __init__(self, threshold: float = 3.5) -> None:
        self.threshold = threshold

    def detect(self, window: np.ndarray) -> tuple[float, bool]:
        """Score the last value using MAD.

        Returns
        -------
        tuple
            ``(modified_z_score, is_anomaly)``

        """
        context = window[:-1]
        value = window[-1]
        median = float(np.median(context))
        mad = float(np.median(np.abs(context - median))) + 1e-10
        # Modified z-score: 0.6745 is the 0.75th quantile of the standard normal
        score = 0.6745 * abs(value - median) / mad
        return float(score), bool(score > self.threshold)


class ConsensusAgent:
    """Aggregate detection decisions from multiple agents.

    Parameters
    ----------
    method
        Voting method: ``"majority"``, ``"any"``, or ``"weighted"``.
    weights
        Per-agent weights for weighted voting. Required when method is
        ``"weighted"``.

    """

    def __init__(
        self,
        method: str = "majority",
        weights: list[float] | None = None,
    ) -> None:
        self.method = method
        self.weights = weights

    def decide(self, flags: list[bool], scores: list[float]) -> bool:  # noqa: ARG002
        """Aggregate agent flags into a final decision.

        Parameters
        ----------
        flags
            Per-agent anomaly flags.
        scores
            Per-agent anomaly scores (used for weighted voting).

        Returns
        -------
        bool
            Final anomaly decision.

        """
        if self.method == "any":
            return any(flags)
        if self.method == "weighted" and self.weights is not None:
            weighted_sum = sum(w for w, f in zip(self.weights, flags, strict=False) if f)
            return weighted_sum >= 0.5 * sum(self.weights)
        # majority (default)
        return sum(flags) > len(flags) / 2
