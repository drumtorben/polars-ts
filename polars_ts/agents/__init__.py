"""Agentic forecasting framework (TimeSeriesScientist-style).

Multi-agent pipeline that automates data diagnostics, model selection,
forecasting, and reporting.  Closes #156.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "LLMBackend": ("polars_ts.agents._protocol", "LLMBackend"),
    "RuleBasedBackend": ("polars_ts.agents._protocol", "RuleBasedBackend"),
    "AgentContext": ("polars_ts.agents._protocol", "AgentContext"),
    "CuratorAgent": ("polars_ts.agents.curator", "CuratorAgent"),
    "CurationReport": ("polars_ts.agents.curator", "CurationReport"),
    "PlannerAgent": ("polars_ts.agents.planner", "PlannerAgent"),
    "ForecastPlan": ("polars_ts.agents.planner", "ForecastPlan"),
    "ForecasterAgent": ("polars_ts.agents.forecaster", "ForecasterAgent"),
    "ForecastAgentResult": ("polars_ts.agents.forecaster", "ForecastAgentResult"),
    "ReporterAgent": ("polars_ts.agents.reporter", "ReporterAgent"),
    "ForecastReport": ("polars_ts.agents.reporter", "ForecastReport"),
    "TimeSeriesScientist": ("polars_ts.agents.scientist", "TimeSeriesScientist"),
    "ScientistResult": ("polars_ts.agents.scientist", "ScientistResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
