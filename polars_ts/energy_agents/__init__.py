"""Hierarchical multi-agent energy and demand forecasting.

A region -> grid -> household hierarchy of demand-forecasting agents with
weather-context adjustment, renewable-intermittency netting, and peak-shaving
demand response, made coherent via the hierarchical reconciliation module.
Closes #162.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "GridHierarchy": ("polars_ts.energy_agents.hierarchy", "GridHierarchy"),
    "DemandForecastAgent": ("polars_ts.energy_agents.agents", "DemandForecastAgent"),
    "WeatherContextAgent": ("polars_ts.energy_agents.agents", "WeatherContextAgent"),
    "RenewableAgent": ("polars_ts.energy_agents.agents", "RenewableAgent"),
    "DemandResponseAgent": ("polars_ts.energy_agents.agents", "DemandResponseAgent"),
    "EnergyGridOrchestrator": ("polars_ts.energy_agents.orchestrator", "EnergyGridOrchestrator"),
    "EnergyForecastResult": ("polars_ts.energy_agents.orchestrator", "EnergyForecastResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
