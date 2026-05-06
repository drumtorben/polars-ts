"""Multi-agent RL framework for portfolio optimization and sequential decisions.

Provides specialized agents (risk, return, allocation) that collaborate
to produce portfolio weights, evaluated in a PortfolioEnv.  Closes #158.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "PortfolioEnv": ("polars_ts.marl.env", "PortfolioEnv"),
    "RiskAgent": ("polars_ts.marl.agents", "RiskAgent"),
    "ReturnAgent": ("polars_ts.marl.agents", "ReturnAgent"),
    "AllocationAgent": ("polars_ts.marl.agents", "AllocationAgent"),
    "MARLOrchestrator": ("polars_ts.marl.orchestrator", "MARLOrchestrator"),
    "MARLResult": ("polars_ts.marl.orchestrator", "MARLResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
