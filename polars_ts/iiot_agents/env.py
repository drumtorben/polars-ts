"""Predictive-maintenance environment for industrial IoT sensor streams.

Models a machine degrading toward failure as a multi-sensor time series. At
each step a maintenance agent chooses to operate, inspect, or maintain; the
reward trades machine uptime against inspection/maintenance cost and a large
penalty for running a degraded machine to failure.

The environment plays back a fixed degradation trajectory (actions do not
mutate the sensor stream); the learning signal is the *timing* of maintenance
relative to the true failure point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Discrete maintenance actions.
OPERATE, INSPECT, MAINTAIN = 0, 1, 2
ACTIONS: tuple[str, ...] = ("operate", "inspect", "maintain")


class MachineEnv:
    """Environment over a machine's multi-sensor degradation trajectory.

    Parameters
    ----------
    sensors
        2D array ``(n_steps, n_sensors)`` of sensor readings (e.g. vibration
        amplitude, temperature, current). May contain ``NaN`` for dropped
        samples (carried forward).
    failure_step
        Index at which the machine fails. If omitted it is inferred from
        ``health`` crossing ``failure_threshold`` (or the last step if never).
    health
        Optional ground-truth health trajectory in ``[0, 1]`` (1 = healthy).
        When omitted, downstream agents estimate health from ``sensors``.
    failure_threshold
        Health level at or below which the machine is considered failed.
    maintenance_cost, inspect_cost, failure_penalty, uptime_reward
        Reward components (see :meth:`step`).

    """

    n_actions: int = 3

    def __init__(
        self,
        sensors: np.ndarray,
        failure_step: int | None = None,
        health: np.ndarray | None = None,
        failure_threshold: float = 0.2,
        maintenance_cost: float = 1.0,
        inspect_cost: float = 0.1,
        failure_penalty: float = 10.0,
        uptime_reward: float = 1.0,
    ) -> None:
        sensors = np.asarray(sensors, dtype=np.float64)
        if sensors.ndim != 2:
            raise ValueError("sensors must be a 2D array of shape (n_steps, n_sensors)")
        self.n_steps, self.n_sensors = sensors.shape
        if self.n_steps < 2:
            raise ValueError("sensors must contain at least two observations")
        self.sensors = _forward_fill(sensors)

        self.health = np.asarray(health, dtype=np.float64) if health is not None else None
        if self.health is not None and self.health.shape != (self.n_steps,):
            raise ValueError("health must be 1D with one entry per step")

        self.failure_threshold = failure_threshold
        self.maintenance_cost = maintenance_cost
        self.inspect_cost = inspect_cost
        self.failure_penalty = failure_penalty
        self.uptime_reward = uptime_reward

        self.failure_step = self._resolve_failure_step(failure_step)
        self._step = 0

    def _resolve_failure_step(self, failure_step: int | None) -> int:
        if failure_step is not None:
            return int(failure_step)
        if self.health is not None:
            below = np.nonzero(self.health <= self.failure_threshold)[0]
            return int(below[0]) if below.size else self.n_steps - 1
        return self.n_steps - 1

    def reset(self) -> np.ndarray:
        """Reset to the first observation and return it."""
        self._step = 0
        return self.sensors[0].copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one observation given a maintenance ``action``.

        Reward logic:

        - ``MAINTAIN`` — pay ``maintenance_cost`` but gain a timeliness bonus
          that peaks just before ``failure_step`` (rewards well-timed
          preventive maintenance; penalises maintaining a healthy machine).
        - ``INSPECT`` — pay a small ``inspect_cost``.
        - ``OPERATE`` — earn ``uptime_reward`` before failure; incur
          ``failure_penalty`` if operated at/after ``failure_step``.

        Returns
        -------
        tuple
            ``(observation, reward, done, info)``.

        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        idx = self._step
        failed = idx >= self.failure_step

        if action == MAINTAIN:
            # Timeliness in [0, 1]: 1 just before failure, decaying earlier.
            timeliness = max(0.0, 1.0 - (self.failure_step - idx) / max(self.failure_step, 1))
            reward = self.failure_penalty * timeliness - self.maintenance_cost
        elif action == INSPECT:
            reward = -self.inspect_cost
        else:  # OPERATE
            reward = -self.failure_penalty if failed else self.uptime_reward

        self._step += 1
        done = self._step >= self.n_steps
        obs = self.sensors[self._step].copy() if not done else np.zeros(self.n_sensors)
        info = {
            "step": idx,
            "action": action,
            "failed": bool(failed),
            "health": float(self.health[idx]) if self.health is not None else None,
        }
        return obs, reward, done, info


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Carry the last valid reading forward per sensor; back-fill leading NaNs."""
    out = arr.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        last = np.nan
        for i in range(col.shape[0]):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
        if np.isnan(col[0]):
            valid = col[~np.isnan(col)]
            col[np.isnan(col)] = valid[0] if valid.size else 0.0
        out[:, c] = col
    return out
