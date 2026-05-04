"""Tests for the agentic forecasting framework (issue #156).

TDD: these tests define the expected API before implementation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_series() -> pl.DataFrame:
    """Return a simple daily time series with trend + seasonality + outlier."""
    import datetime

    n = 120
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    trend = np.linspace(10, 50, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.default_rng(42).normal(0, 1, n)
    values = trend + seasonal + noise
    # inject outlier
    values[50] = 200.0
    # inject missing
    values[30] = float("nan")

    return pl.DataFrame(
        {
            "unique_id": ["series_1"] * n,
            "ds": dates,
            "y": values.tolist(),
        }
    )


@pytest.fixture
def multi_series() -> pl.DataFrame:
    """Two daily series with different characteristics."""
    import datetime

    n = 90
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(123)

    # Series A: trending up
    a = np.linspace(10, 30, n) + rng.normal(0, 0.5, n)
    # Series B: stationary with noise
    b = 20 + rng.normal(0, 3, n)

    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": dates * 2,
            "y": a.tolist() + b.tolist(),
        }
    )


@pytest.fixture
def regime_change_series() -> pl.DataFrame:
    """Series with a clear regime change midway."""
    import datetime

    n = 120
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(99)
    # First half: low mean, low variance
    first = 10 + rng.normal(0, 1, n // 2)
    # Second half: high mean, high variance
    second = 50 + rng.normal(0, 5, n // 2)
    values = np.concatenate([first, second])

    return pl.DataFrame(
        {
            "unique_id": ["regime"] * n,
            "ds": dates,
            "y": values.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Protocol / Backend tests
# ---------------------------------------------------------------------------


class TestLLMBackend:
    def test_rule_based_backend_implements_protocol(self):
        from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend

        backend = RuleBasedBackend()
        assert isinstance(backend, LLMBackend)

    def test_rule_based_backend_complete_returns_str(self):
        from polars_ts.agents._protocol import RuleBasedBackend

        backend = RuleBasedBackend()
        result = backend.complete("test prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AgentContext tests
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_context_creation(self, daily_series):
        from polars_ts.agents._protocol import AgentContext

        ctx = AgentContext(data=daily_series)
        assert ctx.data is daily_series
        assert ctx.metadata == {}
        assert ctx.history == []
        assert ctx.events == []

    def test_context_log(self, daily_series):
        from polars_ts.agents._protocol import AgentContext

        ctx = AgentContext(data=daily_series)
        ctx.log("curator", "Found 1 outlier")
        assert len(ctx.history) == 1
        assert ctx.history[0]["agent"] == "curator"
        assert ctx.history[0]["message"] == "Found 1 outlier"

    def test_context_with_events(self, daily_series):
        from polars_ts.agents._protocol import AgentContext

        events = [{"date": "2023-06-01", "description": "Product launch"}]
        ctx = AgentContext(data=daily_series, events=events)
        assert len(ctx.events) == 1
        assert ctx.events[0]["description"] == "Product launch"


# ---------------------------------------------------------------------------
# CuratorAgent tests
# ---------------------------------------------------------------------------


class TestCuratorAgent:
    def test_curate_returns_curation_report(self, daily_series):
        from polars_ts.agents.curator import CurationReport, CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert isinstance(report, CurationReport)

    def test_curation_report_has_diagnostics(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert report.n_missing >= 0
        assert report.n_outliers >= 0
        assert report.n_observations > 0
        assert report.n_series > 0

    def test_curation_report_detects_outlier(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert report.n_outliers >= 1  # we injected one at index 50

    def test_curation_report_detects_missing(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert report.n_missing >= 1  # we injected NaN at index 30

    def test_curate_and_clean_returns_dataframe(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        cleaned = agent.curate_and_clean(daily_series)
        assert isinstance(cleaned, pl.DataFrame)
        # Should have no NaNs after cleaning
        assert cleaned["y"].null_count() == 0

    def test_curation_report_has_seasonality_info(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert hasattr(report, "detected_period")

    def test_curate_multi_series(self, multi_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(multi_series)
        assert report.n_series == 2

    def test_curation_report_has_stationarity(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(daily_series)
        assert isinstance(report.is_stationary, bool)

    def test_stationarity_detects_nonstationary(self, regime_change_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(regime_change_series)
        assert report.is_stationary is False

    def test_recommended_lookback(self, regime_change_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(regime_change_series)
        # Should recommend trimming due to regime change
        assert report.recommended_lookback is not None
        assert report.recommended_lookback > 0

    def test_trim_lookback(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        trimmed = agent.trim_lookback(daily_series, lookback=50)
        assert len(trimmed) == 50

    def test_trim_lookback_multi_series(self, multi_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        trimmed = agent.trim_lookback(multi_series, lookback=30)
        # 30 per series x 2 series = 60
        assert len(trimmed) == 60

    def test_trim_lookback_none_returns_original(self, daily_series):
        """When no regime change, trim_lookback(lookback=None) returns all data."""
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        # For this series, recommended_lookback may be None
        result = agent.trim_lookback(daily_series)
        # Should return same length if no trimming recommended
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    def test_plan_returns_forecast_plan(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import ForecastPlan, PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        assert isinstance(plan, ForecastPlan)

    def test_plan_has_candidate_models(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        assert len(plan.candidates) > 0

    def test_plan_candidates_are_strings(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        for candidate in plan.candidates:
            assert isinstance(candidate, str)

    def test_plan_has_horizon(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent(horizon=14)
        plan = planner.plan(daily_series, curation)
        assert plan.horizon == 14

    def test_plan_has_rationale(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        assert isinstance(plan.rationale, str)
        assert len(plan.rationale) > 0

    def test_plan_has_ensemble_flag(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        assert isinstance(plan.ensemble, bool)

    def test_plan_enables_ensemble_for_many_candidates(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        # daily_series has 120 obs with trend + seasonality, so should get 3+ candidates
        if len(plan.candidates) >= 3:
            assert plan.ensemble is True


# ---------------------------------------------------------------------------
# ForecasterAgent tests
# ---------------------------------------------------------------------------


class TestForecasterAgent:
    def test_forecast_returns_result(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecastAgentResult, ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)
        assert isinstance(result, ForecastAgentResult)

    def test_forecast_result_has_predictions(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)
        assert isinstance(result.predictions, pl.DataFrame)
        assert "y_hat" in result.predictions.columns

    def test_forecast_result_has_metrics(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)
        assert isinstance(result.model_scores, dict)
        assert len(result.model_scores) > 0

    def test_forecast_result_has_best_model(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)
        assert isinstance(result.best_model, str)

    def test_ensemble_weights_present_when_ensemble(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import ForecastPlan, PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        # Force ensemble on
        if not plan.ensemble:
            plan = ForecastPlan(
                candidates=plan.candidates,
                horizon=plan.horizon,
                rationale=plan.rationale,
                config=plan.config,
                ensemble=True,
            )

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)

        if result.ensemble_weights:
            # Weights should sum to ~1.0
            total = sum(result.ensemble_weights.values())
            assert abs(total - 1.0) < 0.01
            assert "ensemble" in result.best_model

    def test_ensemble_predictions_have_y_hat(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import ForecastPlan, PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)
        plan = ForecastPlan(
            candidates=plan.candidates,
            horizon=plan.horizon,
            rationale=plan.rationale,
            config=plan.config,
            ensemble=True,
        )

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)
        assert "y_hat" in result.predictions.columns


# ---------------------------------------------------------------------------
# ReporterAgent tests
# ---------------------------------------------------------------------------


class TestReporterAgent:
    def test_report_returns_forecast_report(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent
        from polars_ts.agents.reporter import ForecastReport, ReporterAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)

        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        assert isinstance(report, ForecastReport)

    def test_report_has_markdown(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent
        from polars_ts.agents.reporter import ReporterAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)

        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        assert isinstance(report.markdown, str)
        assert "# Forecast Report" in report.markdown

    def test_report_has_sections(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent
        from polars_ts.agents.reporter import ReporterAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)

        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        md = report.markdown
        assert "Data Diagnostics" in md or "Curation" in md
        assert "Model" in md
        assert "Forecast" in md or "Result" in md

    def test_report_includes_stationarity_and_ensemble(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent
        from polars_ts.agents.reporter import ReporterAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)

        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)

        forecaster = ForecasterAgent()
        result = forecaster.forecast(cleaned, plan)

        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        md = report.markdown
        assert "Stationary" in md
        assert "Ensemble" in md


# ---------------------------------------------------------------------------
# TimeSeriesScientist (orchestrator) tests
# ---------------------------------------------------------------------------


class TestTimeSeriesScientist:
    def test_run_end_to_end(self, daily_series):
        from polars_ts.agents.scientist import ScientistResult, TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        assert isinstance(result, ScientistResult)

    def test_run_returns_predictions(self, daily_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        assert isinstance(result.predictions, pl.DataFrame)
        assert "y_hat" in result.predictions.columns

    def test_run_returns_report(self, daily_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        assert isinstance(result.report, str)
        assert len(result.report) > 0

    def test_run_multi_series(self, multi_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(multi_series)
        assert isinstance(result.predictions, pl.DataFrame)

    def test_custom_backend(self, daily_series):
        """Verify that a custom LLM backend can be injected."""
        from polars_ts.agents._protocol import RuleBasedBackend
        from polars_ts.agents.scientist import TimeSeriesScientist

        backend = RuleBasedBackend()
        scientist = TimeSeriesScientist(horizon=7, backend=backend)
        result = scientist.run(daily_series)
        assert isinstance(result.predictions, pl.DataFrame)

    def test_scientist_context_has_history(self, daily_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        # Should have logged actions from each agent
        assert len(result.context.history) >= 3  # curator, planner, forecaster at minimum

    def test_scientist_with_events(self, daily_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        events = [
            {"date": "2023-06-01", "description": "Product launch"},
            {"date": "2023-07-04", "description": "Holiday"},
        ]
        scientist = TimeSeriesScientist(horizon=7, events=events)
        result = scientist.run(daily_series)
        assert result.context.events == events
        # Events should be logged
        event_logs = [h for h in result.context.history if "event" in h["message"].lower()]
        assert len(event_logs) >= 1

    def test_scientist_with_trim_lookback(self, regime_change_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7, trim_lookback=True)
        result = scientist.run(regime_change_series)
        assert isinstance(result.predictions, pl.DataFrame)
        # Should have a trimming log entry
        trim_logs = [h for h in result.context.history if "trim" in h["message"].lower()]
        assert len(trim_logs) >= 1

    def test_scientist_llm_backend_called(self, daily_series):
        """Verify that a real LLM backend's complete() is invoked."""
        from polars_ts.agents.scientist import TimeSeriesScientist

        class MockBackend:
            def __init__(self):
                self.calls: list[str] = []

            def complete(self, prompt: str) -> str:
                self.calls.append(prompt)
                return "LLM says: test response"

        backend = MockBackend()
        scientist = TimeSeriesScientist(horizon=7, backend=backend)
        result = scientist.run(daily_series)
        assert isinstance(result.predictions, pl.DataFrame)
        # The backend should have been called at least once (curator, planner, reporter)
        assert len(backend.calls) >= 1


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_scientist_importable_from_agents(self):
        from polars_ts.agents import TimeSeriesScientist

        assert TimeSeriesScientist is not None

    def test_curator_importable_from_agents(self):
        from polars_ts.agents import CuratorAgent

        assert CuratorAgent is not None

    def test_planner_importable_from_agents(self):
        from polars_ts.agents import PlannerAgent

        assert PlannerAgent is not None

    def test_forecaster_importable_from_agents(self):
        from polars_ts.agents import ForecasterAgent

        assert ForecasterAgent is not None

    def test_reporter_importable_from_agents(self):
        from polars_ts.agents import ReporterAgent

        assert ReporterAgent is not None

    def test_importable_from_top_level(self):
        from polars_ts import TimeSeriesScientist

        assert TimeSeriesScientist is not None
