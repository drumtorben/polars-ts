"""Gymnasium-compatible RL environment for forecast-based decision making."""

from __future__ import annotations

from typing import Any

import numpy as np


class ForecastEnv:
    """A gymnasium-like environment wrapping a polars-ts forecast pipeline.

    At each step, the agent observes recent time series values and a
    forecast, then takes an action (e.g. inventory order, trading signal).
    The reward is computed from a configurable reward function.

    Parameters
    ----------
    data
        Numpy array of shape ``(n_steps,)`` with the actual time series values.
    forecasts
        Numpy array of shape ``(n_steps,)`` with forecast values.
    window_size
        Number of recent observations provided as the observation.
    reward_fn
        Callable ``(action, actual, forecast) -> float``. Defaults to
        negative absolute error: ``-|actual - action|``.

    """

    def __init__(
        self,
        data: np.ndarray,
        forecasts: np.ndarray,
        window_size: int = 10,
        reward_fn: Any | None = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.forecasts = np.asarray(forecasts, dtype=np.float64)
        self.window_size = window_size
        self.reward_fn = reward_fn or (lambda action, actual, _forecast: -abs(actual - action))
        self._step = 0
        self._max_steps = len(self.data) - window_size

        if self._max_steps <= 0:
            raise ValueError("data must be longer than window_size")

    def reset(self) -> np.ndarray:
        """Reset the environment. Return the initial observation."""
        self._step = 0
        return self._get_obs()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Take one step.

        Parameters
        ----------
        action
            The agent's decision for this timestep.

        Returns
        -------
        tuple
            ``(observation, reward, done, info)``

        """
        idx = self.window_size + self._step
        actual = float(self.data[idx])
        forecast = float(self.forecasts[idx])

        reward = float(self.reward_fn(action, actual, forecast))
        self._step += 1
        done = self._step >= self._max_steps

        obs = self._get_obs() if not done else np.zeros(self.window_size + 1)
        info = {"actual": actual, "forecast": forecast}

        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build observation: recent values + current forecast."""
        start = self._step
        end = start + self.window_size
        recent = self.data[start:end]
        forecast = self.forecasts[end] if end < len(self.forecasts) else 0.0
        return np.append(recent, forecast)
