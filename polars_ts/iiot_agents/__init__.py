"""Industrial IoT predictive-maintenance agents.

Combines vibration/sensor spectral analysis, multi-sensor health-index fusion,
Remaining Useful Life (RUL) estimation, and a reinforcement-learning
maintenance scheduler that learns optimal intervention timing (minimising
downtime and maintenance cost). Closes #161.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "MachineEnv": ("polars_ts.iiot_agents.env", "MachineEnv"),
    "SpectralFeatureAgent": ("polars_ts.iiot_agents.agents", "SpectralFeatureAgent"),
    "HealthIndexAgent": ("polars_ts.iiot_agents.agents", "HealthIndexAgent"),
    "RULEstimator": ("polars_ts.iiot_agents.agents", "RULEstimator"),
    "MaintenanceSchedulerAgent": ("polars_ts.iiot_agents.agents", "MaintenanceSchedulerAgent"),
    "MaintenanceOrchestrator": ("polars_ts.iiot_agents.orchestrator", "MaintenanceOrchestrator"),
    "MaintenanceResult": ("polars_ts.iiot_agents.orchestrator", "MaintenanceResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
