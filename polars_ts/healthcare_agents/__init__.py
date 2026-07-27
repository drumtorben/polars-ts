"""Multi-agent clinical decision support for EHR vital-sign time series.

Specialized agents score each vital-sign observation for sepsis risk (qSOFA +
SIRS), physiological derangement, and a NEWS-style escalation tier, while a
contextual-bandit treatment recommender adapts interventions online. Handles
irregularly sampled observations and privacy-preserving federated averaging of
per-site agent parameters. Closes #160.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "ClinicalEnv": ("polars_ts.healthcare_agents.env", "ClinicalEnv"),
    "VITAL_CHANNELS": ("polars_ts.healthcare_agents.env", "VITAL_CHANNELS"),
    "SepsisWarningAgent": ("polars_ts.healthcare_agents.agents", "SepsisWarningAgent"),
    "VitalMonitorAgent": ("polars_ts.healthcare_agents.agents", "VitalMonitorAgent"),
    "EscalationAgent": ("polars_ts.healthcare_agents.agents", "EscalationAgent"),
    "TreatmentAgent": ("polars_ts.healthcare_agents.agents", "TreatmentAgent"),
    "federated_average": ("polars_ts.healthcare_agents.agents", "federated_average"),
    "ClinicalOrchestrator": ("polars_ts.healthcare_agents.orchestrator", "ClinicalOrchestrator"),
    "ClinicalResult": ("polars_ts.healthcare_agents.orchestrator", "ClinicalResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
