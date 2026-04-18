"""Tests for baseline forecast models."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from polars_ts.models.baselines import (
    fft_forecast,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": [date(2024, 1, i) for i in range(1, 7)] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


# ---------- naive_forecast ----------


class TestNaiveForecast:
    def test_basic(self):
        df = _make_df()
        result = naive_forecast(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6  # 3 per series × 2 series

        a = result.filter(pl.col("unique_id") == "A")
        assert a["y_hat"].to_list() == [6.0, 6.0, 6.0]

        b = result.filter(pl.col("unique_id") == "B")
        assert b["y_hat"].to_list() == [60.0, 60.0, 60.0]

    def test_future_dates_correct(self):
        df = _make_df()
        result = naive_forecast(df, h=2)
        a = result.filter(pl.col("unique_id") == "A")
        assert a["ds"].to_list() == [date(2024, 1, 7), date(2024, 1, 8)]

    def test_single_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "y": [10.0, 20.0, 30.0],
            }
        )
        result = naive_forecast(df, h=1)
        assert len(result) == 1
        assert result["y_hat"][0] == 30.0

    def test_invalid_horizon(self):
        df = _make_df()
        with pytest.raises(ValueError, match="positive"):
            naive_forecast(df, h=0)

    def test_unsorted_input(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A", "A"],
                "ds": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
                "y": [30.0, 10.0, 20.0],
            }
        )
        result = naive_forecast(df, h=1)
        # Last value after sorting is 30.0
        assert result["y_hat"][0] == 30.0


# ---------- seasonal_naive_forecast ----------


class TestSeasonalNaiveForecast:
    def test_basic(self):
        df = _make_df()
        result = seasonal_naive_forecast(df, h=3, season_length=3)

        a = result.filter(pl.col("unique_id") == "A")
        # Last 3 values of A: [4, 5, 6] → forecast cycles: [4, 5, 6]
        assert a["y_hat"].to_list() == [4.0, 5.0, 6.0]

    def test_cycles_beyond_season(self):
        df = _make_df()
        result = seasonal_naive_forecast(df, h=5, season_length=3)

        a = result.filter(pl.col("unique_id") == "A")
        # [4, 5, 6, 4, 5]
        assert a["y_hat"].to_list() == [4.0, 5.0, 6.0, 4.0, 5.0]

    def test_season_length_one_equals_naive(self):
        df = _make_df()
        naive = naive_forecast(df, h=3)
        snaive = seasonal_naive_forecast(df, h=3, season_length=1)
        assert naive["y_hat"].to_list() == snaive["y_hat"].to_list()

    def test_invalid_season_length(self):
        df = _make_df()
        with pytest.raises(ValueError, match="positive"):
            seasonal_naive_forecast(df, h=3, season_length=0)

    def test_season_longer_than_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "y": [1.0, 2.0, 3.0],
            }
        )
        # season_length=5 but only 3 values — uses all 3
        result = seasonal_naive_forecast(df, h=4, season_length=5)
        assert result["y_hat"].to_list() == [1.0, 2.0, 3.0, 1.0]


# ---------- moving_average_forecast ----------


class TestMovingAverageForecast:
    def test_basic(self):
        df = _make_df()
        result = moving_average_forecast(df, h=2, window_size=3)

        a = result.filter(pl.col("unique_id") == "A")
        expected_avg = (4.0 + 5.0 + 6.0) / 3
        assert a["y_hat"].to_list() == [expected_avg, expected_avg]

    def test_window_equals_full_series(self):
        df = _make_df()
        result = moving_average_forecast(df, h=1, window_size=6)

        a = result.filter(pl.col("unique_id") == "A")
        expected = sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) / 6
        assert a["y_hat"][0] == pytest.approx(expected)

    def test_window_larger_than_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "y": [3.0, 6.0, 9.0],
            }
        )
        # window_size=10 but only 3 values — averages all
        result = moving_average_forecast(df, h=1, window_size=10)
        assert result["y_hat"][0] == pytest.approx(6.0)

    def test_invalid_window_size(self):
        df = _make_df()
        with pytest.raises(ValueError, match="positive"):
            moving_average_forecast(df, h=1, window_size=0)

    def test_groups_independent(self):
        df = _make_df()
        result = moving_average_forecast(df, h=1, window_size=3)

        a_avg = result.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b_avg = result.filter(pl.col("unique_id") == "B")["y_hat"][0]
        assert a_avg == pytest.approx(5.0)
        assert b_avg == pytest.approx(50.0)


# ---------- fft_forecast ----------


class TestFFTForecast:
    def test_basic(self):
        df = _make_df()
        result = fft_forecast(df, h=3, n_harmonics=2)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_constant_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 6,
                "ds": [date(2024, 1, i) for i in range(1, 7)],
                "y": [5.0] * 6,
            }
        )
        result = fft_forecast(df, h=3, n_harmonics=2)
        # Constant series → forecast should be ~5.0
        for v in result["y_hat"].to_list():
            assert v == pytest.approx(5.0, abs=1e-10)

    def test_periodic_series(self):
        n = 100
        t = np.arange(n, dtype=np.float64)
        y = 10.0 + 3.0 * np.sin(2 * np.pi * t / 10)  # period=10

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * n,
                "ds": [date(2024, 1, 1) + __import__("datetime").timedelta(days=int(i)) for i in t],
                "y": y.tolist(),
            }
        )
        result = fft_forecast(df, h=10, n_harmonics=3)
        # Forecast should roughly follow the sine pattern
        expected = 10.0 + 3.0 * np.sin(2 * np.pi * np.arange(n, n + 10) / 10)
        actual = result["y_hat"].to_numpy()
        np.testing.assert_allclose(actual, expected, atol=0.5)

    def test_invalid_n_harmonics(self):
        df = _make_df()
        with pytest.raises(ValueError, match="positive"):
            fft_forecast(df, h=1, n_harmonics=0)

    def test_future_dates(self):
        df = _make_df()
        result = fft_forecast(df, h=2)
        a = result.filter(pl.col("unique_id") == "A")
        assert a["ds"].to_list() == [date(2024, 1, 7), date(2024, 1, 8)]


# ---------- top-level import ----------


def test_top_level_imports():
    import polars_ts

    assert callable(polars_ts.naive_forecast)
    assert callable(polars_ts.seasonal_naive_forecast)
    assert callable(polars_ts.moving_average_forecast)
    assert callable(polars_ts.fft_forecast)


def test_models_submodule_imports():
    from polars_ts.models import (
        fft_forecast,
        moving_average_forecast,
        naive_forecast,
        seasonal_naive_forecast,
    )

    assert callable(naive_forecast)
    assert callable(seasonal_naive_forecast)
    assert callable(moving_average_forecast)
    assert callable(fft_forecast)
