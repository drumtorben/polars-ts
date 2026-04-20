"""Tests for exponential smoothing forecasters (#49)."""

from datetime import date, timedelta

import polars as pl
import pytest

from polars_ts.models.exponential_smoothing import (
    holt_forecast,
    holt_winters_forecast,
    ses_forecast,
)


def _make_df(n: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) + 10.0 for i in range(n)] + [float(2 * i) + 5.0 for i in range(n)],
        }
    )


class TestSES:
    def test_basic(self):
        result = ses_forecast(_make_df(), h=3, alpha=0.3)
        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_flat_forecast(self):
        result = ses_forecast(_make_df(n=20), h=3, alpha=0.3)
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[0] == pytest.approx(a[1])  # SES produces flat forecast
        assert a[1] == pytest.approx(a[2])

    def test_alpha_bounds(self):
        with pytest.raises(ValueError, match="alpha"):
            ses_forecast(_make_df(), h=1, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ses_forecast(_make_df(), h=1, alpha=1.0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="positive"):
            ses_forecast(_make_df(), h=0)


class TestHolt:
    def test_basic(self):
        result = holt_forecast(_make_df(), h=3)
        assert len(result) == 6

    def test_trend(self):
        result = holt_forecast(_make_df(), h=3, alpha=0.5, beta=0.3)
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[1] > a[0]  # Upward trend

    def test_beta_bounds(self):
        with pytest.raises(ValueError, match="beta"):
            holt_forecast(_make_df(), h=1, beta=0.0)

    def test_short_series(self):
        df = pl.DataFrame({"unique_id": ["A"], "ds": [date(2024, 1, 1)], "y": [1.0]})
        with pytest.raises(ValueError, match="at least 2"):
            holt_forecast(df, h=1)


class TestHoltWinters:
    def _make_seasonal(self, n: int = 48, m: int = 12) -> pl.DataFrame:
        import numpy as np

        base = date(2024, 1, 1)
        values = [10.0 + 0.5 * i + 5.0 * np.sin(2 * np.pi * i / m) for i in range(n)]
        return pl.DataFrame(
            {
                "unique_id": ["A"] * n,
                "ds": [base + timedelta(days=i) for i in range(n)],
                "y": values,
            }
        )

    def test_additive(self):
        result = holt_winters_forecast(self._make_seasonal(), h=12, season_length=12, seasonal="additive")
        assert len(result) == 12

    def test_multiplicative(self):
        df = self._make_seasonal()
        # Shift to strictly positive for multiplicative
        df = df.with_columns((pl.col("y") + 20.0).alias("y"))
        result = holt_winters_forecast(df, h=6, season_length=12, seasonal="multiplicative")
        assert len(result) == 6

    def test_invalid_seasonal(self):
        with pytest.raises(ValueError, match="seasonal"):
            holt_winters_forecast(self._make_seasonal(), h=1, season_length=12, seasonal="invalid")

    def test_short_series(self):
        df = pl.DataFrame(
            {"unique_id": ["A"] * 10, "ds": [date(2024, 1, i + 1) for i in range(10)], "y": list(range(10))}
        )
        with pytest.raises(ValueError, match="2\\*season_length"):
            holt_winters_forecast(df, h=1, season_length=12)

    def test_gamma_bounds(self):
        with pytest.raises(ValueError, match="gamma"):
            holt_winters_forecast(self._make_seasonal(), h=1, season_length=12, gamma=0.0)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.ses_forecast is ses_forecast
    assert polars_ts.holt_forecast is holt_forecast
    assert polars_ts.holt_winters_forecast is holt_winters_forecast
