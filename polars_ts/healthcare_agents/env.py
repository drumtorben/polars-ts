"""Clinical monitoring environment for EHR vital-sign time series.

Models an ICU patient trajectory as a sequence of (possibly irregularly
sampled) vital-sign observations. At each step an agent chooses an escalation
tier; when ground-truth deterioration labels are supplied the reward reflects
early, correct escalation while penalising both alarm fatigue (over-escalation)
and missed deterioration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Canonical vital-sign channel order used across the healthcare_agents module.
VITAL_CHANNELS: tuple[str, ...] = (
    "heart_rate",
    "systolic_bp",
    "respiratory_rate",
    "temperature",
    "spo2",
)


class ClinicalEnv:
    """Gymnasium-like environment over a patient's vital-sign trajectory.

    Parameters
    ----------
    vitals
        2D array ``(n_steps, n_channels)`` of vital-sign observations. Channels
        are interpreted in :data:`VITAL_CHANNELS` order unless ``channels`` is
        given. Missing readings may be encoded as ``NaN`` (carried forward).
    times
        Optional 1D array of observation timestamps (hours since admission) for
        irregularly sampled series. When omitted, unit spacing is assumed. Used
        to expose the elapsed interval since the previous reading in ``info``.
    labels
        Optional ground-truth boolean array (``True`` = clinical deterioration
        at that step). When provided, rewards reflect escalation accuracy.
    channels
        Optional channel names overriding :data:`VITAL_CHANNELS`.

    """

    #: Number of discrete escalation tiers (0 = routine … 3 = ICU/rapid-response).
    n_tiers: int = 4

    def __init__(
        self,
        vitals: np.ndarray,
        times: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        channels: tuple[str, ...] | None = None,
    ) -> None:
        vitals = np.asarray(vitals, dtype=np.float64)
        if vitals.ndim != 2:
            raise ValueError("vitals must be a 2D array of shape (n_steps, n_channels)")
        self.n_steps, self.n_channels = vitals.shape
        if self.n_steps < 1:
            raise ValueError("vitals must contain at least one observation")

        self.channels = channels or VITAL_CHANNELS[: self.n_channels]
        if len(self.channels) != self.n_channels:
            raise ValueError(f"channels length {len(self.channels)} != n_channels {self.n_channels}")

        # Carry-forward imputation for missing (NaN) readings; back-fill any leading NaNs.
        self.vitals = _forward_fill(vitals)

        self.times = (
            np.asarray(times, dtype=np.float64) if times is not None else np.arange(self.n_steps, dtype=np.float64)
        )
        if self.times.shape != (self.n_steps,):
            raise ValueError("times must be 1D with one entry per step")

        self.labels = np.asarray(labels, dtype=bool) if labels is not None else None
        if self.labels is not None and self.labels.shape != (self.n_steps,):
            raise ValueError("labels must be 1D with one entry per step")

        self._step = 0

    def reset(self) -> np.ndarray:
        """Reset to the first observation and return it."""
        self._step = 0
        return self.vitals[0].copy()

    def step(self, tier: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one observation given a chosen escalation ``tier``.

        Parameters
        ----------
        tier
            Chosen escalation tier in ``[0, n_tiers)``.

        Returns
        -------
        tuple
            ``(observation, reward, done, info)``. On the terminal step the
            observation is a zero vector.

        """
        if not 0 <= tier < self.n_tiers:
            raise ValueError(f"tier must be in [0, {self.n_tiers}), got {tier}")

        idx = self._step
        elapsed = float(self.times[idx] - self.times[idx - 1]) if idx > 0 else 0.0
        reward = self._reward(idx, tier)

        self._step += 1
        done = self._step >= self.n_steps
        obs = self.vitals[self._step].copy() if not done else np.zeros(self.n_channels)
        info = {
            "step": idx,
            "time": float(self.times[idx]),
            "elapsed": elapsed,
            "tier": tier,
            "deteriorating": bool(self.labels[idx]) if self.labels is not None else None,
        }
        return obs, reward, done, info

    def _reward(self, idx: int, tier: int) -> float:
        """Escalation reward: reward matched urgency, penalise alarm fatigue and misses."""
        if self.labels is None:
            # Unsupervised: mild penalty per escalation tier to discourage crying wolf.
            return -0.1 * tier
        if self.labels[idx]:
            # Deterioration present: reward proportional to escalation, strong miss penalty.
            return float(tier) if tier > 0 else -2.0
        # Stable patient: reward calm, penalise unnecessary escalation (alarm fatigue).
        return 0.2 if tier == 0 else -0.5 * tier


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Carry the last valid reading forward per channel; back-fill leading NaNs."""
    out = arr.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        last = np.nan
        for i in range(col.shape[0]):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
        # Back-fill any leading NaNs with the first valid value.
        if np.isnan(col[0]):
            valid = col[~np.isnan(col)]
            col[np.isnan(col)] = valid[0] if valid.size else 0.0
        out[:, c] = col
    return out
