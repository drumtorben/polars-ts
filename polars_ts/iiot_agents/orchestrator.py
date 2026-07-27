"""MaintenanceOrchestrator: trains and evaluates maintenance agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polars_ts.iiot_agents.agents import (
    HealthIndexAgent,
    MaintenanceSchedulerAgent,
    RULEstimator,
)
from polars_ts.iiot_agents.env import MAINTAIN, MachineEnv


@dataclass
class MaintenanceResult:
    """Output of a :class:`MaintenanceOrchestrator` run.

    Attributes
    ----------
    actions
        Per-step maintenance action id from the final greedy policy.
    health_index
        Per-step fused health index.
    rul
        Per-step Remaining Useful Life estimate (steps).
    first_maintenance_step
        Index of the first ``MAINTAIN`` action, or ``-1`` if never.
    total_reward
        Total environment reward of the final greedy evaluation pass.
    history
        Per-step diagnostic records.

    """

    actions: np.ndarray
    health_index: np.ndarray
    rul: np.ndarray
    first_maintenance_step: int = -1
    total_reward: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class MaintenanceOrchestrator:
    """Coordinate predictive-maintenance agents over a machine trajectory.

    The scheduler is trained over ``n_episodes`` replays of the trajectory,
    then evaluated once greedily to produce the reported schedule.

    Parameters
    ----------
    n_episodes
        Number of Q-learning training episodes over the trajectory.
    window
        Sliding-window length for spectral/health feature extraction.
    failure_threshold
        Health level defining failure (shared by env, health, and RUL).
    seed
        Seed for the scheduler's RNG.

    """

    def __init__(
        self,
        n_episodes: int = 50,
        window: int = 5,
        failure_threshold: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.n_episodes = n_episodes
        self.window = window
        self.failure_threshold = failure_threshold
        self.seed = seed

    def _health_series(self, sensors: np.ndarray, health: np.ndarray | None) -> np.ndarray:
        """Per-step fused health index (uses ground truth when provided)."""
        if health is not None:
            return np.asarray(health, dtype=np.float64)
        agent = HealthIndexAgent(warmup=self.window)
        agent.fit_baseline(sensors)
        n = len(sensors)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            lo = max(0, i - self.window + 1)
            out[i] = agent.score(sensors[lo : i + 1])
        return out

    def run(
        self,
        sensors: np.ndarray,
        health: np.ndarray | None = None,
        failure_step: int | None = None,
    ) -> MaintenanceResult:
        """Train the scheduler then return its greedy maintenance schedule.

        Parameters
        ----------
        sensors
            2D array ``(n_steps, n_sensors)`` of sensor readings.
        health
            Optional ground-truth health trajectory.
        failure_step
            Optional explicit failure index.

        Returns
        -------
        MaintenanceResult

        """
        health_series = self._health_series(sensors, health)
        scheduler = MaintenanceSchedulerAgent(n_actions=MachineEnv.n_actions, seed=self.seed)
        rul_estimator = RULEstimator(failure_threshold=self.failure_threshold)

        def make_env() -> MachineEnv:
            return MachineEnv(
                sensors,
                failure_step=failure_step,
                health=health_series,
                failure_threshold=self.failure_threshold,
            )

        # --- Training: replay the trajectory, learning Q-values. ---
        for _ in range(self.n_episodes):
            env = make_env()
            env.reset()
            state = scheduler.bucket(float(health_series[0]))
            done = False
            while not done:
                action = scheduler.act(state, explore=True)
                _, reward, done, info = env.step(action)
                nxt = info["step"] + 1
                next_state = scheduler.bucket(float(health_series[min(nxt, len(health_series) - 1)]))
                scheduler.update(state, action, reward, next_state)
                state = next_state

        # --- Evaluation: single greedy pass. ---
        env = make_env()
        env.reset()
        actions: list[int] = []
        rul: list[float] = []
        history: list[dict[str, Any]] = []
        total_reward = 0.0
        first_maint = -1
        done = False
        idx = 0
        while not done:
            hb = scheduler.bucket(float(health_series[idx]))
            action = scheduler.act(hb, explore=False)
            rul_est = rul_estimator.estimate(health_series[: idx + 1])
            _, reward, done, info = env.step(action)
            total_reward += reward
            actions.append(action)
            rul.append(rul_est)
            if action == MAINTAIN and first_maint < 0:
                first_maint = idx
            history.append(
                {
                    "step": idx,
                    "health": float(health_series[idx]),
                    "rul": rul_est,
                    "action": action,
                    "reward": reward,
                }
            )
            idx += 1

        return MaintenanceResult(
            actions=np.array(actions, dtype=np.int64),
            health_index=health_series.copy(),
            rul=np.array(rul, dtype=np.float64),
            first_maintenance_step=first_maint,
            total_reward=total_reward,
            history=history,
        )
