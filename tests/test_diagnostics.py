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


def test_constant_series_acf():
    """Constant series should have ACF=1 at lag 0, ACF=0 for all other lags."""
    df = pl.DataFrame({"unique_id": ["A"] * 50, "ds": [date(2024, 1, 1)] * 50, "y": [5.0] * 50})
    result = acf(df, max_lags=5)
    lag0 = result.filter(pl.col("lag") == 0)["acf"][0]
    assert lag0 == pytest.approx(1.0)
    for lag in range(1, 6):
        val = result.filter(pl.col("lag") == lag)["acf"][0]
        assert val == pytest.approx(0.0, abs=1e-10)


def test_max_lags_exceeds_data():
    """When max_lags > n, should compute up to n-1 lags without error."""
    df = pl.DataFrame({"unique_id": ["A"] * 10, "ds": [date(2024, 1, 1)] * 10, "y": list(range(10))})
    result = acf(df, max_lags=100)
    # Should have at most 10 lags (0 through 9)
    assert result["lag"].max() <= 9


def test_ar3_pacf_significant_at_lag3():
    """AR(3) process should show significant PACF at lag 3."""
    rng = np.random.default_rng(123)
    n = 500
    y = [0.0, 0.0, 0.0]
    for _ in range(n - 3):
        y.append(0.3 * y[-1] + 0.2 * y[-2] + 0.2 * y[-3] + rng.normal(0, 0.5))
    df = pl.DataFrame({"unique_id": ["A"] * n, "ds": [date(2024, 1, 1)] * n, "y": y})
    result = pacf(df, max_lags=5)
    lag3 = result.filter(pl.col("lag") == 3)["pacf"][0]
    # PACF at lag 3 should be non-negligible for AR(3)
    assert abs(lag3) > 0.05


def test_custom_column_names_acf():
    """ACF should work with non-default column names."""
    df = pl.DataFrame({"series": ["X"] * 50, "value": np.random.default_rng(42).normal(0, 1, 50).tolist()})
    result = acf(df, target_col="value", id_col="series", max_lags=5)
    assert "series" in result.columns
    assert result["lag"].max() == 5


def test_top_level_imports():
    import polars_ts

    assert polars_ts.acf is acf
    assert polars_ts.pacf is pacf
