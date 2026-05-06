"""RL-based autonomous anomaly detection agents.

Multi-agent framework where specialized detectors (z-score, rolling std, MAD)
independently score observations, then a consensus agent aggregates their
decisions via majority vote, any-vote, or weighted combination.  Closes #159.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "AnomalyEnv": ("polars_ts.anomaly_agents.env", "AnomalyEnv"),
    "ZScoreAgent": ("polars_ts.anomaly_agents.agents", "ZScoreAgent"),
    "RollingStdAgent": ("polars_ts.anomaly_agents.agents", "RollingStdAgent"),
    "MADAgent": ("polars_ts.anomaly_agents.agents", "MADAgent"),
    "ConsensusAgent": ("polars_ts.anomaly_agents.agents", "ConsensusAgent"),
    "AnomalyOrchestrator": ("polars_ts.anomaly_agents.orchestrator", "AnomalyOrchestrator"),
    "AnomalyResult": ("polars_ts.anomaly_agents.orchestrator", "AnomalyResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
