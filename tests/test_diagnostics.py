"""Tests for residual diagnostics (#57)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from polars_ts.diagnostics import acf, pacf


def _make_white_noise(n: int = 100, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [date(2024, 1, 1)] * n,  # dates not used by acf/pacf
            "y": rng.normal(0, 1, n).tolist(),
        }
    )


def _make_ar1(n: int = 200, phi: float = 0.8, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = [0.0]
    for _ in range(n - 1):
        y.append(phi * y[-1] + rng.normal(0, 0.5))
    return pl.DataFrame({"unique_id": ["A"] * n, "ds": [date(2024, 1, 1)] * n, "y": y})


class TestACF:
    def test_output_columns(self):
        result = acf(_make_white_noise(), max_lags=10)
        assert set(result.columns) == {"unique_id", "lag", "acf", "ci_lower", "ci_upper"}

    def test_lag_zero_is_one(self):
        result = acf(_make_white_noise(), max_lags=5)
        lag0 = result.filter(pl.col("lag") == 0)["acf"][0]
        assert lag0 == pytest.approx(1.0)

    def test_white_noise_within_ci(self):
        result = acf(_make_white_noise(n=200), max_lags=20)
        non_zero = result.filter(pl.col("lag") > 0)
        within = non_zero.filter((pl.col("acf") > pl.col("ci_lower")) & (pl.col("acf") < pl.col("ci_upper")))
        # Most lags should be within CI for white noise
        assert len(within) >= len(non_zero) * 0.8

    def test_ar1_significant_lag1(self):
        result = acf(_make_ar1(phi=0.8), max_lags=5)
        lag1 = result.filter(pl.col("lag") == 1)["acf"][0]
        assert abs(lag1) > 0.5  # Strong autocorrelation

    def test_max_lags_respected(self):
        result = acf(_make_white_noise(), max_lags=5)
        assert result["lag"].max() == 5

    def test_invalid_max_lags(self):
        with pytest.raises(ValueError, match="max_lags"):
            acf(_make_white_noise(), max_lags=0)

    def test_multiple_series(self):
        df = pl.concat(
            [
                _make_white_noise().with_columns(pl.lit("A").alias("unique_id")),
                _make_white_noise(seed=99).with_columns(pl.lit("B").alias("unique_id")),
            ]
        )
        result = acf(df, max_lags=5)
        assert len(result["unique_id"].unique()) == 2


class TestPACF:
    def test_output_columns(self):
        result = pacf(_make_white_noise(), max_lags=10)
        assert set(result.columns) == {"unique_id", "lag", "pacf", "ci_lower", "ci_upper"}

    def test_lag_zero_is_one(self):
        result = pacf(_make_white_noise(), max_lags=5)
        lag0 = result.filter(pl.col("lag") == 0)["pacf"][0]
        assert lag0 == pytest.approx(1.0)

    def test_ar1_significant_lag1_only(self):
        result = pacf(_make_ar1(phi=0.8, n=300), max_lags=10)
        lag1 = result.filter(pl.col("lag") == 1)["pacf"][0]
        lag2 = result.filter(pl.col("lag") == 2)["pacf"][0]
        assert abs(lag1) > 0.5  # PACF at lag 1 should be significant
        assert abs(lag2) < abs(lag1)  # PACF at lag 2 should be smaller


class TestLjungBox:
    def test_basic(self):
        pytest.importorskip("scipy")
        from polars_ts.diagnostics import ljung_box

        result = ljung_box(_make_white_noise(n=100))
        assert "q_stat" in result.columns
        assert "p_value" in result.columns

    def test_white_noise_not_significant(self):
        pytest.importorskip("scipy")
        from polars_ts.diagnostics import ljung_box

        result = ljung_box(_make_white_noise(n=200), lags=[10])
        p = result["p_value"][0]
        assert p > 0.05  # White noise should not be significant

    def test_ar1_significant(self):
        pytest.importorskip("scipy")
        from polars_ts.diagnostics import ljung_box

        result = ljung_box(_make_ar1(phi=0.8, n=200), lags=[10])
        p = result["p_value"][0]
        assert p < 0.05  # AR(1) should be significant


def test_top_level_imports():
    import polars_ts

    assert polars_ts.acf is acf
    assert polars_ts.pacf is pacf
