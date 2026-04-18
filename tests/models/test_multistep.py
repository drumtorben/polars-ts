"""Tests for multi-step forecasting strategies."""

from datetime import date

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402

from polars_ts.models.multistep import DirectForecaster, RecursiveForecaster  # noqa: E402


def _make_linear_df() -> pl.DataFrame:
    """Two series with perfect linear trends: A = t, B = 2t."""
    n = 20
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) for i in range(n)] + [float(2 * i) for i in range(n)],
        }
    )


# ---------- RecursiveForecaster ----------


class TestRecursiveForecaster:
    def test_fit_predict_basic(self):
        df = _make_linear_df()
        fc = RecursiveForecaster(LinearRegression(), lags=[1, 2])
        fc.fit(df)
        result = fc.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6  # 3 steps × 2 series

    def test_linear_trend_accuracy(self):
        df = _make_linear_df()
        fc = RecursiveForecaster(LinearRegression(), lags=[1, 2])
        fc.fit(df)
        result = fc.predict(df, h=3)

        # Series A: y = t, last value is 19, next should be ~20, 21, 22
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[0] == pytest.approx(20.0, abs=0.5)
        assert a[1] == pytest.approx(21.0, abs=1.0)
        assert a[2] == pytest.approx(22.0, abs=1.5)

    def test_multiple_series_independent(self):
        df = _make_linear_df()
        fc = RecursiveForecaster(LinearRegression(), lags=[1, 2])
        fc.fit(df)
        result = fc.predict(df, h=1)

        a_pred = result.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b_pred = result.filter(pl.col("unique_id") == "B")["y_hat"][0]
        # B has double the slope of A
        assert b_pred == pytest.approx(a_pred * 2, abs=1.0)

    def test_future_dates_correct(self):
        df = _make_linear_df()
        fc = RecursiveForecaster(LinearRegression(), lags=[1])
        fc.fit(df)
        result = fc.predict(df, h=2)

        a = result.filter(pl.col("unique_id") == "A")
        assert a["ds"].to_list() == [date(2024, 1, 21), date(2024, 1, 22)]

    def test_invalid_horizon(self):
        df = _make_linear_df()
        fc = RecursiveForecaster(LinearRegression(), lags=[1])
        fc.fit(df)
        with pytest.raises(ValueError, match="positive"):
            fc.predict(df, h=0)

    def test_predict_before_fit(self):
        fc = RecursiveForecaster(LinearRegression(), lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            fc.predict(_make_linear_df(), h=1)

    def test_invalid_lags(self):
        with pytest.raises(ValueError, match="positive"):
            RecursiveForecaster(LinearRegression(), lags=[])

        with pytest.raises(ValueError, match="positive"):
            RecursiveForecaster(LinearRegression(), lags=[-1, 1])

    def test_single_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 10,
                "ds": [date(2024, 1, i + 1) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        fc = RecursiveForecaster(LinearRegression(), lags=[1, 2])
        fc.fit(df)
        result = fc.predict(df, h=2)
        assert len(result) == 2
        assert result["y_hat"][0] == pytest.approx(10.0, abs=0.5)


# ---------- DirectForecaster ----------


class TestDirectForecaster:
    def test_fit_predict_basic(self):
        df = _make_linear_df()
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=3)
        fc.fit(df)
        result = fc.predict(df)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6  # 3 steps × 2 series

    def test_h_models_fitted(self):
        df = _make_linear_df()
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=5)
        fc.fit(df)
        assert len(fc.estimators_) == 5

    def test_linear_trend_accuracy(self):
        df = _make_linear_df()
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=3)
        fc.fit(df)
        result = fc.predict(df)

        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[0] == pytest.approx(20.0, abs=0.5)
        assert a[1] == pytest.approx(21.0, abs=1.0)
        assert a[2] == pytest.approx(22.0, abs=1.5)

    def test_multiple_series_independent(self):
        df = _make_linear_df()
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=1)
        fc.fit(df)
        result = fc.predict(df)

        a_pred = result.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b_pred = result.filter(pl.col("unique_id") == "B")["y_hat"][0]
        assert b_pred == pytest.approx(a_pred * 2, abs=1.0)

    def test_future_dates_correct(self):
        df = _make_linear_df()
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1], h=2)
        fc.fit(df)
        result = fc.predict(df)

        a = result.filter(pl.col("unique_id") == "A")
        assert a["ds"].to_list() == [date(2024, 1, 21), date(2024, 1, 22)]

    def test_predict_before_fit(self):
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1], h=1)
        with pytest.raises(RuntimeError, match="fit"):
            fc.predict(_make_linear_df())

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="positive"):
            DirectForecaster(lambda: LinearRegression(), lags=[1], h=0)

    def test_invalid_lags(self):
        with pytest.raises(ValueError, match="positive"):
            DirectForecaster(lambda: LinearRegression(), lags=[], h=1)

    def test_single_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 10,
                "ds": [date(2024, 1, i + 1) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=2)
        fc.fit(df)
        result = fc.predict(df)
        assert len(result) == 2
        assert result["y_hat"][0] == pytest.approx(10.0, abs=0.5)


# ---------- top-level imports ----------


def test_top_level_imports():
    import polars_ts

    assert polars_ts.RecursiveForecaster is RecursiveForecaster
    assert polars_ts.DirectForecaster is DirectForecaster


def test_models_submodule_imports():
    from polars_ts.models import DirectForecaster, RecursiveForecaster

    assert callable(RecursiveForecaster)
    assert callable(DirectForecaster)
