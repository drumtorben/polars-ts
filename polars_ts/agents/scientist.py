"""TimeSeriesScientist: orchestrator that chains all agents end-to-end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from polars_ts.agents._protocol import AgentContext, LLMBackend, RuleBasedBackend
from polars_ts.agents.curator import CuratorAgent
from polars_ts.agents.forecaster import ForecasterAgent
from polars_ts.agents.planner import PlannerAgent
from polars_ts.agents.reporter import ReporterAgent


@dataclass
class ScientistResult:
    """Full output of a TimeSeriesScientist run."""

    predictions: pl.DataFrame
    report: str
    context: AgentContext


class TimeSeriesScientist:
    """Orchestrates the full agentic forecasting pipeline.

    Chains Curator -> Planner -> Forecaster -> Reporter to automate
    the complete time series analysis workflow.

    Parameters
    ----------
    horizon
        Forecast horizon (number of steps ahead).
    backend
        LLM backend shared across all agents. Defaults to rule-based.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.
    events
        Optional list of event dicts with ``date`` and ``description`` keys
        that provide context for the forecast (e.g., upcoming promotions,
        holidays, or known disruptions).
    trim_lookback
        Whether to automatically trim series to the recommended lookback
        window before forecasting. Defaults to False.

    """

    def __init__(
        self,
        horizon: int = 10,
        backend: LLMBackend | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        events: list[dict[str, Any]] | None = None,
        trim_lookback: bool = False,
    ) -> None:
        self.horizon = horizon
        self.backend = backend or RuleBasedBackend()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.events = events or []
        self.trim_lookback = trim_lookback

    def run(self, df: pl.DataFrame) -> ScientistResult:
        """Execute the full pipeline: curate -> plan -> forecast -> report."""
        ctx = AgentContext(data=df, events=self.events)

        # 1. Curate
        curator = CuratorAgent(
            backend=self.backend,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        curation = curator.curate(df)
        ctx.log("curator", curation.summary)
        cleaned = curator.curate_and_clean(df)

        # Optionally trim lookback window
        if self.trim_lookback and curation.recommended_lookback is not None:
            cleaned = curator.trim_lookback(cleaned, curation.recommended_lookback)
            ctx.log("curator", f"Trimmed to lookback={curation.recommended_lookback}")

        # Log events if provided
        if self.events:
            ctx.log("curator", f"{len(self.events)} event(s) registered as context")

        # 2. Plan
        planner = PlannerAgent(
            backend=self.backend,
            horizon=self.horizon,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        plan = planner.plan(cleaned, curation)
        ctx.log("planner", f"Selected {len(plan.candidates)} candidates: {', '.join(plan.candidates)}")
        if plan.ensemble:
            ctx.log("planner", "Ensemble mode enabled")

        # 3. Forecast
        forecaster = ForecasterAgent(
            backend=self.backend,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        result = forecaster.forecast(cleaned, plan)
        ctx.log(
            "forecaster",
            f"Best model: {result.best_model} (MAE={result.model_scores.get(result.best_model, float('nan')):.4f})",
        )
        if result.ensemble_weights:
            weights_str = ", ".join(f"{k}={v:.3f}" for k, v in result.ensemble_weights.items())
            ctx.log("forecaster", f"Ensemble weights: {weights_str}")

        # 4. Report
        reporter = ReporterAgent(backend=self.backend)
        report = reporter.report(curation, plan, result)
        ctx.log("reporter", "Report generated")

        return ScientistResult(
            predictions=result.predictions,
            report=report.markdown,
            context=ctx,
        )
