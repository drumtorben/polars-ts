"""ReporterAgent: synthesize forecasting process into a transparent report."""

from __future__ import annotations

from dataclasses import dataclass

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend
from polars_ts.agents.curator import CurationReport
from polars_ts.agents.forecaster import ForecastAgentResult
from polars_ts.agents.planner import ForecastPlan


@dataclass
class ForecastReport:
    """Structured report from the forecasting pipeline."""

    markdown: str


class ReporterAgent:
    """Generates a structured report from the full pipeline output.

    Parameters
    ----------
    backend
        LLM backend for narrative generation. Defaults to rule-based templates.

    """

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend = backend or RuleBasedBackend()

    def report(
        self,
        curation: CurationReport,
        plan: ForecastPlan,
        result: ForecastAgentResult,
    ) -> ForecastReport:
        """Generate a markdown report from all pipeline stages."""
        sections: list[str] = []

        sections.append("# Forecast Report\n")

        # Data Diagnostics
        sections.append("## Data Diagnostics\n")
        sections.append(f"- **Series**: {curation.n_series}")
        sections.append(f"- **Observations**: {curation.n_observations}")
        sections.append(f"- **Missing values**: {curation.n_missing}")
        sections.append(f"- **Outliers detected**: {curation.n_outliers}")
        if curation.detected_period:
            sections.append(f"- **Detected period**: {curation.detected_period}")
        sections.append(f"- **Trend**: {'Yes' if curation.has_trend else 'No'}")
        sections.append(f"- **Stationary**: {'Yes' if curation.is_stationary else 'No'}")
        if curation.recommended_lookback:
            sections.append(f"- **Recommended lookback**: {curation.recommended_lookback}")
        sections.append("")

        # Model Selection
        sections.append("## Model Selection\n")
        sections.append(f"- **Candidates**: {', '.join(plan.candidates)}")
        sections.append(f"- **Horizon**: {plan.horizon}")
        sections.append(f"- **Ensemble**: {'Yes' if plan.ensemble else 'No'}")
        sections.append(f"- **Rationale**: {plan.rationale}")
        sections.append("")

        # Forecast Results
        sections.append("## Forecast Results\n")
        sections.append(f"- **Best model**: {result.best_model}")
        sections.append("- **Model scores (MAE)**:")
        for name, score in sorted(result.model_scores.items()):
            sections.append(f"  - {name}: {score:.4f}")
        if result.ensemble_weights:
            sections.append("- **Ensemble weights**:")
            for name, weight in sorted(result.ensemble_weights.items()):
                sections.append(f"  - {name}: {weight:.3f}")
        sections.append(f"- **Prediction rows**: {len(result.predictions)}")
        sections.append("")

        md = "\n".join(sections)

        # Enhance with LLM narrative if available
        if not isinstance(self.backend, RuleBasedBackend):
            llm_narrative = self.backend.complete(f"Write a brief executive summary for this forecast report:\n{md}")
            if llm_narrative:
                md = f"## Executive Summary\n\n{llm_narrative}\n\n{md}"

        return ForecastReport(markdown=md)
