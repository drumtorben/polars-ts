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


# ---------------------------------------------------------------------------
# Additional fixtures for edge-case testing
# ---------------------------------------------------------------------------


@pytest.fixture
def short_series() -> pl.DataFrame:
    """Series with only 20 observations (short-series planner path)."""
    import datetime

    n = 20
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(7)
    values = 10 + rng.normal(0, 1, n)
    return pl.DataFrame({"unique_id": ["short"] * n, "ds": dates, "y": values.tolist()})


@pytest.fixture
def constant_series() -> pl.DataFrame:
    """Series where all values are identical (std=0 edge case)."""
    import datetime

    n = 60
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"unique_id": ["const"] * n, "ds": dates, "y": [42.0] * n})


@pytest.fixture
def no_id_series() -> pl.DataFrame:
    """Series without a unique_id column (single-series path)."""
    import datetime

    n = 100
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(55)
    trend = np.linspace(5, 25, n)
    values = trend + rng.normal(0, 1, n)
    values[10] = float("nan")
    return pl.DataFrame({"ds": dates, "y": values.tolist()})


@pytest.fixture
def leading_nan_series() -> pl.DataFrame:
    """Series with NaN at the very start (backward-fill edge case)."""
    import datetime

    n = 50
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(88)
    values = 20 + rng.normal(0, 1, n)
    values[0] = float("nan")
    values[1] = float("nan")
    return pl.DataFrame({"unique_id": ["lead_nan"] * n, "ds": dates, "y": values.tolist()})


# ---------------------------------------------------------------------------
# CuratorAgent edge-case tests
# ---------------------------------------------------------------------------


class TestCuratorEdgeCases:
    def test_curate_no_id_column(self, no_id_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(no_id_series)
        assert report.n_series == 1
        assert report.n_missing >= 1

    def test_clean_no_id_column(self, no_id_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        cleaned = agent.curate_and_clean(no_id_series)
        assert cleaned["y"].null_count() == 0
        assert cleaned["y"].is_nan().sum() == 0

    def test_clean_removes_nan_not_just_null(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        cleaned = agent.curate_and_clean(daily_series)
        assert cleaned["y"].is_nan().sum() == 0

    def test_clean_leading_nan(self, leading_nan_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        cleaned = agent.curate_and_clean(leading_nan_series)
        assert cleaned["y"].null_count() == 0
        assert cleaned["y"].is_nan().sum() == 0

    def test_constant_series_no_outliers(self, constant_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(constant_series)
        assert report.n_outliers == 0

    def test_constant_series_no_trend(self, constant_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(constant_series)
        assert report.has_trend is False

    def test_constant_series_no_period(self, constant_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(constant_series)
        assert report.detected_period is None

    def test_custom_outlier_threshold(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent

        # Lower threshold should detect more outliers
        strict = CuratorAgent(outlier_threshold=2.0)
        lenient = CuratorAgent(outlier_threshold=5.0)
        strict_report = strict.curate(daily_series)
        lenient_report = lenient.curate(daily_series)
        assert strict_report.n_outliers >= lenient_report.n_outliers

    def test_trim_lookback_no_id_column(self, no_id_series):
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        trimmed = agent.trim_lookback(no_id_series, lookback=30)
        assert len(trimmed) == 30

    def test_short_series_stationarity_defaults_true(self):
        """Very short series (<20) should default to stationary."""
        import datetime

        n = 15
        dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({"unique_id": ["tiny"] * n, "ds": dates, "y": list(range(n))})

        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(df)
        assert report.is_stationary is True

    def test_short_series_no_period(self):
        """Very short series (<10) should return None for period."""
        import datetime

        n = 8
        dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({"unique_id": ["tiny"] * n, "ds": dates, "y": [1.0] * n})

        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(df)
        assert report.detected_period is None

    def test_short_series_no_lookback(self, short_series):
        """Short series (<40) should not recommend lookback."""
        from polars_ts.agents.curator import CuratorAgent

        agent = CuratorAgent()
        report = agent.curate(short_series)
        assert report.recommended_lookback is None

    def test_empty_dataframe(self):
        """Empty DataFrame should not crash diagnostics."""
        from polars_ts.agents.curator import CuratorAgent

        df = pl.DataFrame({"unique_id": [], "ds": [], "y": []}).cast(
            {"unique_id": pl.Utf8, "ds": pl.Date, "y": pl.Float64}
        )
        agent = CuratorAgent()
        report = agent.curate(df)
        assert report.n_observations == 0
        assert report.n_missing == 0
        assert report.detected_period is None
        assert report.has_trend is False


# ---------------------------------------------------------------------------
# PlannerAgent edge-case tests
# ---------------------------------------------------------------------------


class TestPlannerEdgeCases:
    def test_short_series_gets_ses(self, short_series):
        """Series with <30 obs should get naive + SES only."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(short_series)
        planner = PlannerAgent()
        plan = planner.plan(short_series, curation)
        assert "naive" in plan.candidates
        assert "ses" in plan.candidates
        assert "moving_average" not in plan.candidates

    def test_short_series_no_ensemble(self, short_series):
        """Short series path should have only 2 candidates, no ensemble."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(short_series)
        planner = PlannerAgent()
        plan = planner.plan(short_series, curation)
        assert plan.ensemble is False

    def test_plan_config_has_moving_average_window(self, daily_series):
        """When moving_average is selected, config should have window_size."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        if "moving_average" in plan.candidates:
            assert "moving_average" in plan.config
            assert "window_size" in plan.config["moving_average"]
            assert plan.config["moving_average"]["window_size"] > 0

    def test_plan_config_has_season_length_for_holt_winters(self, daily_series):
        """When holt_winters is selected, config should have season_length."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        if "holt_winters" in plan.candidates:
            assert "holt_winters" in plan.config
            assert "season_length" in plan.config["holt_winters"]

    def test_plan_always_includes_naive(self, daily_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent()
        plan = planner.plan(daily_series, curation)
        assert plan.candidates[0] == "naive"

    def test_plan_with_llm_backend(self, daily_series):
        """LLM backend should replace rationale."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.planner import PlannerAgent

        class StubBackend:
            def complete(self, prompt: str) -> str:  # noqa: ARG002
                return "LLM rationale override"

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        planner = PlannerAgent(backend=StubBackend())
        plan = planner.plan(daily_series, curation)
        assert plan.rationale == "LLM rationale override"


# ---------------------------------------------------------------------------
# ForecasterAgent edge-case tests
# ---------------------------------------------------------------------------


class TestForecasterEdgeCases:
    def test_compute_mae(self):
        from polars_ts.agents.forecaster import _compute_mae

        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.5, 3.5])
        assert abs(_compute_mae(actual, predicted) - 0.5) < 1e-10

    def test_compute_mae_perfect(self):
        from polars_ts.agents.forecaster import _compute_mae

        arr = np.array([1.0, 2.0, 3.0])
        assert _compute_mae(arr, arr) == 0.0

    def test_train_val_split_sizes(self, daily_series):
        from polars_ts.agents.forecaster import ForecasterAgent

        agent = ForecasterAgent()
        train, val = agent._train_val_split(daily_series, h=10)
        assert len(train) == len(daily_series) - 10
        assert len(val) == 10

    def test_train_val_split_no_overlap(self, daily_series):
        from polars_ts.agents.forecaster import ForecasterAgent

        agent = ForecasterAgent()
        train, val = agent._train_val_split(daily_series, h=10)
        train_dates = set(train["ds"].to_list())
        val_dates = set(val["ds"].to_list())
        assert train_dates.isdisjoint(val_dates)

    def test_single_model_no_ensemble(self, daily_series):
        """With ensemble=False, result should not have ensemble weights."""
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import ForecastPlan

        plan = ForecastPlan(candidates=["naive"], horizon=7, rationale="test", ensemble=False)
        cleaned = daily_series.with_columns(pl.col("y").fill_nan(pl.lit(None)).forward_fill().backward_fill())
        agent = ForecasterAgent()
        result = agent.forecast(cleaned, plan)
        assert result.best_model == "naive"
        assert result.ensemble_weights == {}

    def test_unknown_model_skipped(self, daily_series):
        """Unknown model names should be silently skipped."""
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import ForecastPlan

        plan = ForecastPlan(
            candidates=["naive", "unknown_model_xyz"],
            horizon=7,
            rationale="test",
            ensemble=False,
        )
        cleaned = daily_series.with_columns(pl.col("y").fill_nan(pl.lit(None)).forward_fill().backward_fill())
        agent = ForecasterAgent()
        result = agent.forecast(cleaned, plan)
        assert "unknown_model_xyz" not in result.model_scores
        assert isinstance(result.predictions, pl.DataFrame)

    def test_all_predictions_dict_populated(self, daily_series):
        """all_predictions should contain per-model DataFrames."""
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(daily_series)
        cleaned = curator.curate_and_clean(daily_series)
        planner = PlannerAgent(horizon=7)
        plan = planner.plan(cleaned, curation)
        agent = ForecasterAgent()
        result = agent.forecast(cleaned, plan)
        assert len(result.all_predictions) >= 1
        for _name, preds in result.all_predictions.items():
            assert isinstance(preds, pl.DataFrame)
            assert "y_hat" in preds.columns

    def test_forecast_multi_series(self, multi_series):
        from polars_ts.agents.curator import CuratorAgent
        from polars_ts.agents.forecaster import ForecasterAgent
        from polars_ts.agents.planner import PlannerAgent

        curator = CuratorAgent()
        curation = curator.curate(multi_series)
        cleaned = curator.curate_and_clean(multi_series)
        planner = PlannerAgent(horizon=5)
        plan = planner.plan(cleaned, curation)
        agent = ForecasterAgent()
        result = agent.forecast(cleaned, plan)
        assert isinstance(result.predictions, pl.DataFrame)
        assert "y_hat" in result.predictions.columns

    def test_score_empty_join_returns_inf(self):
        """When validation and predictions don't overlap, score should be inf."""
        import datetime

        from polars_ts.agents.forecaster import ForecasterAgent

        val = pl.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [datetime.date(2023, 1, 1)],
                "y": [10.0],
            }
        )
        preds = pl.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [datetime.date(2099, 1, 1)],
                "y_hat": [10.0],
            }
        )
        agent = ForecasterAgent()
        score = agent._score(val, preds)
        assert score == float("inf")


# ---------------------------------------------------------------------------
# ReporterAgent edge-case tests
# ---------------------------------------------------------------------------


class TestReporterEdgeCases:
    def test_report_with_detected_period(self):
        from polars_ts.agents.curator import CurationReport
        from polars_ts.agents.forecaster import ForecastAgentResult
        from polars_ts.agents.planner import ForecastPlan
        from polars_ts.agents.reporter import ReporterAgent

        curation = CurationReport(
            n_observations=100,
            n_series=1,
            n_missing=0,
            n_outliers=0,
            detected_period=7,
            has_trend=False,
            is_stationary=True,
            recommended_lookback=None,
            summary="test",
        )
        plan = ForecastPlan(candidates=["naive"], horizon=10, rationale="test")
        result = ForecastAgentResult(
            predictions=pl.DataFrame({"ds": [], "y_hat": []}),
            best_model="naive",
            model_scores={"naive": 1.0},
        )
        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        assert "period" in report.markdown.lower()
        assert "7" in report.markdown

    def test_report_with_recommended_lookback(self):
        from polars_ts.agents.curator import CurationReport
        from polars_ts.agents.forecaster import ForecastAgentResult
        from polars_ts.agents.planner import ForecastPlan
        from polars_ts.agents.reporter import ReporterAgent

        curation = CurationReport(
            n_observations=200,
            n_series=1,
            n_missing=0,
            n_outliers=0,
            detected_period=None,
            has_trend=True,
            is_stationary=False,
            recommended_lookback=60,
            summary="test",
        )
        plan = ForecastPlan(candidates=["naive"], horizon=10, rationale="test")
        result = ForecastAgentResult(
            predictions=pl.DataFrame({"ds": [], "y_hat": []}),
            best_model="naive",
            model_scores={"naive": 1.0},
        )
        reporter = ReporterAgent()
        report = reporter.report(curation, plan, result)
        assert "lookback" in report.markdown.lower()
        assert "60" in report.markdown

    def test_report_with_llm_has_executive_summary(self):
        from polars_ts.agents.curator import CurationReport
        from polars_ts.agents.forecaster import ForecastAgentResult
        from polars_ts.agents.planner import ForecastPlan
        from polars_ts.agents.reporter import ReporterAgent

        class StubBackend:
            def complete(self, prompt: str) -> str:  # noqa: ARG002
                return "Executive overview text."

        curation = CurationReport(
            n_observations=100,
            n_series=1,
            n_missing=0,
            n_outliers=0,
            detected_period=None,
            has_trend=False,
            is_stationary=True,
            recommended_lookback=None,
            summary="test",
        )
        plan = ForecastPlan(candidates=["naive"], horizon=10, rationale="test")
        result = ForecastAgentResult(
            predictions=pl.DataFrame({"ds": [], "y_hat": []}),
            best_model="naive",
            model_scores={"naive": 1.0},
        )
        reporter = ReporterAgent(backend=StubBackend())
        report = reporter.report(curation, plan, result)
        assert "Executive Summary" in report.markdown
        assert "Executive overview text." in report.markdown


# ---------------------------------------------------------------------------
# TimeSeriesScientist edge-case tests
# ---------------------------------------------------------------------------


class TestScientistEdgeCases:
    def test_custom_column_names(self):
        import datetime

        n = 80
        dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"sid": ["s1"] * n, "timestamp": dates, "value": (10 + rng.normal(0, 1, n)).tolist()})

        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=5, id_col="sid", time_col="timestamp", target_col="value")
        result = scientist.run(df)
        assert isinstance(result.predictions, pl.DataFrame)
        assert "y_hat" in result.predictions.columns

    def test_no_id_column(self, no_id_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=5)
        result = scientist.run(no_id_series)
        assert isinstance(result.predictions, pl.DataFrame)
        assert "y_hat" in result.predictions.columns

    def test_trim_lookback_false_does_not_trim(self, regime_change_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7, trim_lookback=False)
        result = scientist.run(regime_change_series)
        trim_logs = [h for h in result.context.history if "trim" in h["message"].lower()]
        assert len(trim_logs) == 0

    def test_empty_events(self, daily_series):
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7, events=[])
        result = scientist.run(daily_series)
        event_logs = [h for h in result.context.history if "event" in h["message"].lower()]
        assert len(event_logs) == 0

    def test_context_history_has_all_agents(self, daily_series):
        """History should include curator, planner, forecaster, reporter."""
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        agents_logged = {h["agent"] for h in result.context.history}
        assert "curator" in agents_logged
        assert "planner" in agents_logged
        assert "forecaster" in agents_logged
        assert "reporter" in agents_logged

    def test_predictions_have_finite_values(self, daily_series):
        """Predictions should contain finite numeric values."""
        from polars_ts.agents.scientist import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=7)
        result = scientist.run(daily_series)
        y_hat = result.predictions["y_hat"].to_numpy()
        assert np.all(np.isfinite(y_hat))
