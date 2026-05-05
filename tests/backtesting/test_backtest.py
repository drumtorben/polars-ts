"""Tests for the unified backtesting framework."""

from datetime import datetime, timedelta

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402

from polars_ts.backtesting.backtest import backtest, compare_models  # noqa: E402
from polars_ts.metrics.forecast import mae, rmse  # noqa: E402
from polars_ts.pipeline import ForecastPipeline  # noqa: E402
from polars_ts.validation.splits import expanding_window_cv  # noqa: E402


def _make_ts(n: int = 40, n_series: int = 2) -> pl.DataFrame:
    rows: list[dict] = []
    base = datetime(2024, 1, 1)
    for s in range(n_series):
        sid = chr(65 + s)
        for i in range(n):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(hours=i),
                    "y": float(i * (s + 1)) + 10.0,
                }
            )
    return pl.DataFrame(rows)


class TestBacktest:
    def test_basic_backtest(self):
        df = _make_ts()
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=3, horizon=2, step=2)
        metrics = {"mae": mae, "rmse": rmse}

        result = backtest(model, cv, metrics, h=2)

        assert "fold_scores" in result
        assert "summary" in result

        fold_scores = result["fold_scores"]
        assert len(fold_scores) == 3
        assert "fold" in fold_scores.columns
        assert "mae" in fold_scores.columns
        assert "rmse" in fold_scores.columns

        summary = result["summary"]
        assert len(summary) == 2
        assert set(summary["metric"].to_list()) == {"mae", "rmse"}
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_inferred_horizon(self):
        df = _make_ts()
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=2, horizon=3, step=2)

        result = backtest(model, cv, {"mae": mae})
        assert len(result["fold_scores"]) == 2

    def test_return_predictions(self):
        df = _make_ts()
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=2, horizon=2, step=2)

        result = backtest(model, cv, {"mae": mae}, h=2, return_predictions=True)

        assert "predictions" in result
        preds = result["predictions"]
        assert "fold" in preds.columns
        assert "y_hat" in preds.columns
        # 2 folds × 2 horizon × 2 series = 8 rows
        assert len(preds) == 8

    def test_per_horizon(self):
        df = _make_ts()
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=2, horizon=3, step=2)

        result = backtest(model, cv, {"mae": mae, "rmse": rmse}, h=3, per_horizon=True)

        assert "per_horizon" in result
        ph = result["per_horizon"]
        assert "horizon_step" in ph.columns
        assert "mae_mean" in ph.columns
        assert "rmse_std" in ph.columns
        assert len(ph) == 3  # 3 horizon steps

    def test_invalid_model_raises(self):
        df = _make_ts()
        cv = expanding_window_cv(df, n_splits=2, horizon=2)
        with pytest.raises(TypeError, match="fit.*predict"):
            backtest("not_a_model", cv, {"mae": mae})

    def test_empty_metrics_raises(self):
        df = _make_ts()
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=2, horizon=2)
        with pytest.raises(ValueError, match="non-empty"):
            backtest(model, cv, {})

    def test_single_series(self):
        df = _make_ts(n_series=1)
        model = ForecastPipeline(LinearRegression(), lags=[1, 2])
        cv = expanding_window_cv(df, n_splits=2, horizon=2, step=2)

        result = backtest(model, cv, {"mae": mae}, h=2)
        assert len(result["fold_scores"]) == 2


class TestCompareModels:
    def test_basic_comparison(self):
        df = _make_ts()
        models = {
            "lr_lag1": ForecastPipeline(LinearRegression(), lags=[1]),
            "lr_lag12": ForecastPipeline(LinearRegression(), lags=[1, 2]),
        }
        cv_kwargs = {"n_splits": 2, "horizon": 2, "step": 2}

        result = compare_models(
            models=models,
            df=df,
            cv=expanding_window_cv,
            cv_kwargs=cv_kwargs,
            metrics={"mae": mae, "rmse": rmse},
            h=2,
        )

        assert "comparison" in result
        assert "fold_scores" in result

        comp = result["comparison"]
        assert len(comp) == 2
        assert set(comp["model"].to_list()) == {"lr_lag1", "lr_lag12"}
        assert "mae_mean" in comp.columns
        assert "rmse_std" in comp.columns

        fold_scores = result["fold_scores"]
        assert "model" in fold_scores.columns
        assert len(fold_scores) == 4  # 2 models × 2 folds

    def test_empty_models_raises(self):
        df = _make_ts()
        with pytest.raises(ValueError, match="non-empty"):
            compare_models(
                models={},
                df=df,
                cv=expanding_window_cv,
                cv_kwargs={"n_splits": 2, "horizon": 2},
                metrics={"mae": mae},
            )
