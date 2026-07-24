"""ClinicalOrchestrator: chains clinical agents over a ClinicalEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polars_ts.healthcare_agents.agents import (
    EscalationAgent,
    SepsisWarningAgent,
    TreatmentAgent,
    VitalMonitorAgent,
)
from polars_ts.healthcare_agents.env import ClinicalEnv


@dataclass
class ClinicalResult:
    """Output of a :class:`ClinicalOrchestrator` run.

    Attributes
    ----------
    escalation_tiers
        Per-step chosen escalation tier.
    sepsis_flags
        Per-step sepsis-risk flag from :class:`SepsisWarningAgent`.
    treatments
        Per-step recommended intervention label.
    total_reward
        Sum of environment rewards (meaningful only when labels are supplied).
    history
        Per-step diagnostic records.

    """

    escalation_tiers: np.ndarray
    sepsis_flags: np.ndarray
    treatments: list[str]
    total_reward: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class ClinicalOrchestrator:
    """Coordinate clinical decision-support agents over a vital-sign trajectory.

    At each observation:

    1. :class:`SepsisWarningAgent` scores sepsis risk (qSOFA + SIRS).
    2. :class:`VitalMonitorAgent` counts out-of-range vitals.
    3. :class:`EscalationAgent` selects an escalation tier (NEWS-style).
    4. :class:`TreatmentAgent` recommends an intervention and, when labels are
       available, learns online from the environment reward.

    Parameters
    ----------
    sepsis_threshold
        Combined qSOFA+SIRS score at which sepsis risk is flagged.
    explore
        Epsilon-greedy exploration rate for the treatment recommender.
    seed
        Seed for the treatment recommender's RNG.

    """

    def __init__(
        self,
        sepsis_threshold: int = 2,
        explore: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.sepsis_threshold = sepsis_threshold
        self.explore = explore
        self.seed = seed

    def run(
        self,
        vitals: np.ndarray,
        times: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> ClinicalResult:
        """Run the multi-agent clinical monitoring loop.

        Parameters
        ----------
        vitals
            2D array ``(n_steps, n_channels)`` of vital-sign observations.
        times
            Optional observation timestamps for irregularly sampled series.
        labels
            Optional ground-truth deterioration labels for reward/learning.

        Returns
        -------
        ClinicalResult

        """
        env = ClinicalEnv(vitals, times=times, labels=labels)
        sepsis_agent = SepsisWarningAgent(threshold=self.sepsis_threshold)
        monitor = VitalMonitorAgent()
        escalation = EscalationAgent(n_tiers=env.n_tiers)
        treatment = TreatmentAgent(seed=self.seed)

        tiers: list[int] = []
        sepsis_flags: list[bool] = []
        treatments: list[str] = []
        history: list[dict[str, Any]] = []
        total_reward = 0.0

        obs = env.reset()
        done = False
        while not done:
            risk, sepsis_flag = sepsis_agent.score(obs)
            n_deranged, _ = monitor.score(obs)
            tier = escalation.decide(obs, sepsis_flag, n_deranged)
            action = treatment.recommend(tier, explore=self.explore)

            obs, reward, done, info = env.step(tier)
            treatment.update(tier, action, reward)
            total_reward += reward

            tiers.append(tier)
            sepsis_flags.append(sepsis_flag)
            treatments.append(treatment.actions[action])
            history.append(
                {
                    "step": info["step"],
                    "time": info["time"],
                    "elapsed": info["elapsed"],
                    "sepsis_risk": risk,
                    "n_deranged": n_deranged,
                    "tier": tier,
                    "treatment": treatment.actions[action],
                    "reward": reward,
                }
            )

        return ClinicalResult(
            escalation_tiers=np.array(tiers, dtype=np.int64),
            sepsis_flags=np.array(sepsis_flags, dtype=bool),
            treatments=treatments,
            total_reward=total_reward,
            history=history,
        )
