"""Tests for VAR model and Granger causality (#50)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.var_model import VARResult, granger_causality, var_fit, var_forecast


def _make_var_data(n: int = 100, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = rng.normal()
    y[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(0, 0.5)
        y[t] = 0.2 * x[t - 1] + 0.6 * y[t - 1] + rng.normal(0, 0.5)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "ds": [base + timedelta(days=i) for i in range(n)],
            "x": x.tolist(),
            "y": y.tolist(),
        }
    )


class TestVAR:
    def test_fit_basic(self):
        df = _make_var_data()
        result = var_fit(df, target_cols=["x", "y"], p=1)
        assert isinstance(result, VARResult)
        assert result.p == 1
        assert result.coefficients.shape == (2, 3)  # 2 vars, 2*1 lags + 1 intercept

    def test_fit_p2(self):
        df = _make_var_data()
        result = var_fit(df, target_cols=["x", "y"], p=2)
        assert result.coefficients.shape == (2, 5)  # 2 vars, 2*2 lags + 1 intercept

    def test_forecast_basic(self):
        df = _make_var_data()
        model = var_fit(df, target_cols=["x", "y"], p=1)
        fc = var_forecast(model, horizon=5)
        assert "x" in fc.columns
        assert "y" in fc.columns
        assert len(fc) == 5

    def test_forecast_reasonable(self):
        df = _make_var_data(n=200)
        model = var_fit(df, target_cols=["x", "y"], p=1)
        fc = var_forecast(model, horizon=3)
        # Forecasts should be in a reasonable range
        for col in ["x", "y"]:
            vals = fc[col].to_list()
            assert all(abs(v) < 50 for v in vals)

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p must"):
            var_fit(_make_var_data(), target_cols=["x", "y"], p=0)

    def test_single_col_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            var_fit(_make_var_data(), target_cols=["x"])

    def test_invalid_horizon(self):
        model = var_fit(_make_var_data(), target_cols=["x", "y"], p=1)
        with pytest.raises(ValueError, match="positive"):
            var_forecast(model, horizon=0)

    def test_residuals_stored(self):
        model = var_fit(_make_var_data(), target_cols=["x", "y"], p=1)
        assert model.residuals.shape[1] == 2


class TestGrangerCausality:
    def test_basic(self):
        pytest.importorskip("scipy")
        df = _make_var_data()
        result = granger_causality(df, cause_col="x", effect_col="y", max_lag=3)
        assert "f_stat" in result.columns
        assert "p_value" in result.columns
        assert len(result) == 3

    def test_significant_causality(self):
        pytest.importorskip("scipy")
        df = _make_var_data(n=300)
        result = granger_causality(df, cause_col="x", effect_col="y", max_lag=3)
        # x does Granger-cause y in our DGP (coefficient 0.2)
        assert result["p_value"][0] < 0.1


def test_var_three_variables():
    """VAR should work with 3+ variables."""
    rng = np.random.default_rng(42)
    n = 100
    base = date(2024, 1, 1)
    df = pl.DataFrame(
        {
            "ds": [base + timedelta(days=i) for i in range(n)],
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
            "z": rng.normal(0, 1, n).tolist(),
        }
    )
    model = var_fit(df, target_cols=["x", "y", "z"], p=1)
    assert model.coefficients.shape[0] == 3
    fc = var_forecast(model, horizon=3)
    assert set(fc.columns) >= {"x", "y", "z"}


def test_var_custom_time_col():
    """VAR should accept custom time column name."""
    rng = np.random.default_rng(42)
    n = 50
    base = date(2024, 1, 1)
    df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
        }
    )
    model = var_fit(df, target_cols=["x", "y"], p=1, time_col="timestamp")
    fc = var_forecast(model, horizon=3)
    assert len(fc) == 3


def test_granger_no_causality():
    """Independent series should show no Granger causality."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(42)
    n = 200
    base = date(2024, 1, 1)
    df = pl.DataFrame(
        {
            "ds": [base + timedelta(days=i) for i in range(n)],
            "x": rng.normal(0, 1, n).tolist(),
            "y": rng.normal(0, 1, n).tolist(),
        }
    )
    result = granger_causality(df, cause_col="x", effect_col="y", max_lag=3)
    # Independent series should have high p-values
    assert result["p_value"].min() > 0.01


def test_top_level_imports():
    import polars_ts

    assert polars_ts.var_fit is var_fit
    assert polars_ts.var_forecast is var_forecast
