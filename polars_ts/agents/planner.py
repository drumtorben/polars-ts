"""PlannerAgent: multi-modal diagnostics for model selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend
from polars_ts.agents.curator import CurationReport


@dataclass
class ForecastPlan:
    """Model selection plan produced by the PlannerAgent."""

    candidates: list[str]
    horizon: int
    rationale: str
    config: dict[str, dict] = field(default_factory=dict)
    ensemble: bool = False


class PlannerAgent:
    """Selects candidate models based on data characteristics.

    Parameters
    ----------
    backend
        LLM backend for guided selection. Defaults to rule-based heuristics.
    horizon
        Forecast horizon (number of steps ahead).
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.

    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        horizon: int = 10,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.horizon = horizon
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def plan(self, df: pl.DataFrame, curation: CurationReport) -> ForecastPlan:  # noqa: ARG002
        """Produce a forecast plan based on data diagnostics."""
        candidates: list[str] = []
        rationale_parts: list[str] = []
        config: dict[str, dict] = {}

        n = curation.n_observations // max(curation.n_series, 1)

        # Always include naive as baseline
        candidates.append("naive")
        rationale_parts.append("Naive baseline for comparison.")

        # Short series: simple methods
        if n < 30:
            candidates.append("ses")
            rationale_parts.append("Short series — SES is robust.")
        else:
            # Moving average for any reasonable-length series
            window_size = min(max(n // 10, 3), 30)
            candidates.append("moving_average")
            config["moving_average"] = {"window_size": window_size}
            rationale_parts.append(f"Moving average (window={window_size}) for smoothed baseline.")

            if curation.has_trend:
                candidates.append("holt")
                rationale_parts.append("Trend detected — Holt's method appropriate.")

            if curation.detected_period is not None:
                candidates.append("holt_winters")
                config["holt_winters"] = {"season_length": curation.detected_period}
                rationale_parts.append(
                    f"Seasonality (period={curation.detected_period}) — Holt-Winters for trend+seasonal."
                )

            if n >= 50:
                candidates.append("ses")
                rationale_parts.append("SES as additional candidate for longer series.")

        # Enable ensemble when 3+ candidates available
        use_ensemble = len(candidates) >= 3
        if use_ensemble:
            rationale_parts.append("3+ candidates — ensemble enabled for adaptive combination.")

        # Enhance rationale with LLM if available
        rationale = " ".join(rationale_parts)
        if not isinstance(self.backend, RuleBasedBackend):
            llm_response = self.backend.complete(
                f"Given these time series characteristics:\n"
                f"- {n} observations per series, {curation.n_series} series\n"
                f"- Trend: {curation.has_trend}, Period: {curation.detected_period}\n"
                f"- Stationary: {curation.is_stationary}\n"
                f"Selected candidates: {candidates}\n"
                f"Provide a brief rationale for this model selection."
            )
            if llm_response:
                rationale = llm_response

        return ForecastPlan(
            candidates=candidates,
            horizon=self.horizon,
            rationale=rationale,
            config=config,
            ensemble=use_ensemble,
        )
