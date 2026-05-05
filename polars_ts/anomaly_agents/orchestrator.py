"""AnomalyOrchestrator: chains detection agents over an AnomalyEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polars_ts.anomaly_agents.agents import ConsensusAgent, MADAgent, RollingStdAgent, ZScoreAgent
from polars_ts.anomaly_agents.env import AnomalyEnv


@dataclass
class AnomalyResult:
    """Output of an AnomalyOrchestrator run."""

    detections: np.ndarray
    agent_scores: dict[str, np.ndarray]
    history: list[dict[str, Any]] = field(default_factory=list)


class AnomalyOrchestrator:
    """Orchestrates anomaly detection agents over a time series.

    At each step:
    1. ZScoreAgent scores the current observation.
    2. RollingStdAgent scores via median-deviation.
    3. MADAgent scores via robust MAD.
    4. ConsensusAgent aggregates via majority vote.

    Parameters
    ----------
    window_size
        Observation window for the environment and agents.
    consensus_method
        Voting method for ConsensusAgent.
    z_threshold
        Z-score threshold for ZScoreAgent.
    std_threshold
        Threshold for RollingStdAgent.
    mad_threshold
        Threshold for MADAgent.

    """

    def __init__(
        self,
        window_size: int = 20,
        consensus_method: str = "majority",
        z_threshold: float = 3.0,
        std_threshold: float = 3.0,
        mad_threshold: float = 3.5,
    ) -> None:
        self.window_size = window_size
        self.consensus_method = consensus_method
        self.z_threshold = z_threshold
        self.std_threshold = std_threshold
        self.mad_threshold = mad_threshold

    def run(self, data: np.ndarray, labels: np.ndarray | None = None) -> AnomalyResult:
        """Run the multi-agent anomaly detection loop.

        Parameters
        ----------
        data
            1D array of time series values.
        labels
            Optional ground-truth labels for reward computation.

        Returns
        -------
        AnomalyResult

        """
        env = AnomalyEnv(data, window_size=self.window_size, labels=labels)

        z_agent = ZScoreAgent(threshold=self.z_threshold)
        std_agent = RollingStdAgent(threshold=self.std_threshold)
        mad_agent = MADAgent(threshold=self.mad_threshold)
        consensus = ConsensusAgent(method=self.consensus_method)

        history: list[dict[str, Any]] = []
        detections: list[bool] = []
        z_scores: list[float] = []
        std_scores: list[float] = []
        mad_scores: list[float] = []

        obs = env.reset()
        done = False

        while not done:
            z_score, z_flag = z_agent.detect(obs)
            std_score, std_flag = std_agent.detect(obs)
            mad_score, mad_flag = mad_agent.detect(obs)

            flags = [z_flag, std_flag, mad_flag]
            scores = [z_score, std_score, mad_score]
            decision = consensus.decide(flags, scores)

            obs, _reward, done, info = env.step(decision)

            detections.append(decision)
            z_scores.append(z_score)
            std_scores.append(std_score)
            mad_scores.append(mad_score)
            history.append(
                {
                    "step": env._step - 1,
                    "value": info["value"],
                    "z_score": z_score,
                    "std_score": std_score,
                    "mad_score": mad_score,
                    "decision": decision,
                }
            )

        return AnomalyResult(
            detections=np.array(detections, dtype=bool),
            agent_scores={
                "z_score": np.array(z_scores),
                "rolling_std": np.array(std_scores),
                "mad": np.array(mad_scores),
            },
            history=history,
        )
