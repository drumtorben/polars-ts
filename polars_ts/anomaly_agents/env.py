"""Anomaly detection environment for RL-based adaptive thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np


class AnomalyEnv:
    """Gymnasium-like environment for anomaly detection.

    At each step, the agent observes a window of recent values and decides
    whether the current point is anomalous. If ground-truth labels are
    provided, rewards reflect detection accuracy.

    Parameters
    ----------
    data
        1D array of time series values.
    window_size
        Number of recent observations in each observation.
    labels
        Optional ground-truth boolean array (True = anomaly). When provided,
        correct detections get positive reward and false alarms negative.
        Without labels, reward is based on statistical deviation.

    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 20,
        labels: np.ndarray | None = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.window_size = window_size
        self.labels = np.asarray(labels, dtype=bool) if labels is not None else None
        self._step = 0
        self._max_steps = len(self.data) - window_size

        if self._max_steps <= 0:
            raise ValueError("data must be longer than window_size")

    def reset(self) -> np.ndarray:
        """Reset and return initial observation."""
        self._step = 0
        return self._get_obs()

    def step(self, action: bool) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Take one step.

        Parameters
        ----------
        action
            True to flag the current point as anomalous.

        Returns
        -------
        tuple
            ``(observation, reward, done, info)``

        """
        idx = self.window_size + self._step
        value = float(self.data[idx])

        # Compute adaptive threshold from window
        window = self.data[self._step : idx]
        mean = float(window.mean())
        std = float(window.std()) + 1e-10
        z_score = abs(value - mean) / std

        # Reward
        if self.labels is not None:
            is_anomaly = bool(self.labels[idx])
            if action and is_anomaly:
                reward = 1.0  # true positive
            elif action and not is_anomaly:
                reward = -0.5  # false positive
            elif not action and is_anomaly:
                reward = -1.0  # false negative
            else:
                reward = 0.1  # true negative
        else:
            # Unsupervised: reward based on consistency with z-score
            if action and z_score > 3.0:
                reward = z_score  # reward proportional to deviation
            elif action:
                reward = -1.0  # flagged but not extreme
            else:
                reward = 0.0

        self._step += 1
        done = self._step >= self._max_steps

        obs = self._get_obs() if not done else np.zeros(self.window_size)
        info = {"value": value, "threshold": mean + 3 * std, "z_score": z_score}

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build observation: window of recent values."""
        start = self._step
        end = start + self.window_size
        return self.data[start:end].copy()
