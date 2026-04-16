"""Tests for forecast error metrics (MAE, RMSE, MAPE, sMAPE, MASE, CRPS)."""

import math

import polars as pl
import pytest

from polars_ts.metrics.forecast import crps, mae, mape, mase, rmse, smape


# --- Fixtures ---


@pytest.fixture
def perfect_forecast():
    """Actual equals predicted."""
    return pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "y_hat": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def simple_forecast():
    """Simple forecast with known errors."""
    return pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "y_hat": [1.5, 2.5, 2.5, 3.5]})


@pytest.fixture
def grouped_forecast():
    """Two series with different error levels."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "ds": list(range(4)) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "y_hat": [1.0, 2.0, 3.0, 4.0, 12.0, 22.0, 28.0, 38.0],
        }
    )


# --- MAE ---


class TestMAE:
    def test_perfect_forecast(self, perfect_forecast):
        assert mae(perfect_forecast) == 0.0

    def test_simple_values(self, simple_forecast):
        # errors: |0.5| + |0.5| + |0.5| + |0.5| = 2.0, mean = 0.5
        assert mae(simple_forecast) == 0.5

    def test_symmetric(self):
        df = pl.DataFrame({"y": [1.0, 3.0], "y_hat": [3.0, 1.0]})
        assert mae(df) == 2.0

    def test_per_group(self, grouped_forecast):
        result = mae(grouped_forecast, id_col="unique_id")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2
        a_mae = result.filter(pl.col("unique_id") == "A")["mae"].item()
        b_mae = result.filter(pl.col("unique_id") == "B")["mae"].item()
        assert a_mae == 0.0
        assert b_mae == 2.0

    def test_custom_columns(self):
        df = pl.DataFrame({"actual": [1.0, 2.0], "pred": [1.5, 2.5]})
        assert mae(df, actual_col="actual", predicted_col="pred") == 0.5

    def test_returns_float(self, simple_forecast):
        result = mae(simple_forecast)
        assert isinstance(result, float)


# --- RMSE ---


class TestRMSE:
    def test_perfect_forecast(self, perfect_forecast):
        assert rmse(perfect_forecast) == 0.0

    def test_simple_values(self, simple_forecast):
        # errors^2: 0.25 * 4 = 1.0, mean = 0.25, sqrt = 0.5
        assert rmse(simple_forecast) == 0.5

    def test_penalizes_large_errors(self):
        # Same MAE (1.0), but RMSE differs
        df_uniform = pl.DataFrame({"y": [0.0, 0.0], "y_hat": [1.0, 1.0]})
        df_spike = pl.DataFrame({"y": [0.0, 0.0], "y_hat": [0.0, 2.0]})
        assert rmse(df_spike) > rmse(df_uniform)

    def test_per_group(self, grouped_forecast):
        result = rmse(grouped_forecast, id_col="unique_id")
        a_rmse = result.filter(pl.col("unique_id") == "A")["rmse"].item()
        assert a_rmse == 0.0

    def test_returns_float(self, simple_forecast):
        assert isinstance(rmse(simple_forecast), float)


# --- MAPE ---


class TestMAPE:
    def test_perfect_forecast(self, perfect_forecast):
        assert mape(perfect_forecast) == 0.0

    def test_simple_values(self):
        df = pl.DataFrame({"y": [100.0, 200.0], "y_hat": [110.0, 180.0]})
        # |10/100| + |20/200| = 0.1 + 0.1 = 0.2, mean = 0.1
        assert abs(mape(df) - 0.1) < 1e-10

    def test_excludes_zero_actuals(self):
        df = pl.DataFrame({"y": [0.0, 100.0], "y_hat": [10.0, 110.0]})
        # Only the second row counts: |10/100| = 0.1
        assert abs(mape(df) - 0.1) < 1e-10

    def test_per_group(self, grouped_forecast):
        result = mape(grouped_forecast, id_col="unique_id")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2

    def test_returns_float(self):
        df = pl.DataFrame({"y": [100.0], "y_hat": [110.0]})
        assert isinstance(mape(df), float)


# --- sMAPE ---


class TestSMAPE:
    def test_perfect_forecast(self, perfect_forecast):
        assert smape(perfect_forecast) == 0.0

    def test_symmetric_property(self):
        df1 = pl.DataFrame({"y": [100.0], "y_hat": [110.0]})
        df2 = pl.DataFrame({"y": [110.0], "y_hat": [100.0]})
        assert abs(smape(df1) - smape(df2)) < 1e-10

    def test_excludes_double_zero(self):
        df = pl.DataFrame({"y": [0.0, 100.0], "y_hat": [0.0, 110.0]})
        # Only second row: 2*10 / (100+110) = 20/210
        expected = 20.0 / 210.0
        assert abs(smape(df) - expected) < 1e-10

    def test_range(self, simple_forecast):
        score = smape(simple_forecast)
        assert 0.0 <= score <= 2.0

    def test_per_group(self, grouped_forecast):
        result = smape(grouped_forecast, id_col="unique_id")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2


# --- MASE ---


class TestMASE:
    def test_perfect_forecast(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": list(range(5)),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_hat": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = mase(df)
        assert result == 0.0

    def test_naive_baseline_equals_one(self):
        """A forecast that matches the naive baseline should have MASE ~ 1."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": list(range(5)),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_hat": [0.0, 1.0, 2.0, 3.0, 4.0],  # shift by 1 = naive forecast
            }
        )
        result = mase(df, season_length=1)
        assert abs(result - 1.0) < 1e-10

    def test_better_than_naive(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": list(range(5)),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_hat": [1.1, 2.1, 2.9, 3.9, 5.1],
            }
        )
        result = mase(df, season_length=1)
        assert result < 1.0

    def test_per_group(self, grouped_forecast):
        result = mase(grouped_forecast)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2
        assert "mase" in result.columns

    def test_seasonal_period(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8,
                "ds": list(range(8)),
                "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "y_hat": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            }
        )
        result = mase(df, season_length=2)
        assert result == 0.0


# --- CRPS ---


class TestCRPS:
    def test_perfect_quantiles(self):
        df = pl.DataFrame(
            {
                "y": [10.0, 20.0],
                "q_0.1": [10.0, 20.0],
                "q_0.5": [10.0, 20.0],
                "q_0.9": [10.0, 20.0],
            }
        )
        assert crps(df) == 0.0

    def test_auto_detect_columns(self):
        df = pl.DataFrame(
            {
                "y": [10.0],
                "q_0.1": [8.0],
                "q_0.5": [10.0],
                "q_0.9": [12.0],
            }
        )
        score = crps(df)
        assert score > 0.0

    def test_explicit_columns(self):
        df = pl.DataFrame(
            {
                "y": [10.0],
                "low": [8.0],
                "mid": [10.0],
                "high": [12.0],
            }
        )
        score = crps(
            df,
            quantile_cols=["low", "mid", "high"],
            quantiles=[0.1, 0.5, 0.9],
        )
        assert score > 0.0

    def test_wider_intervals_higher_crps(self):
        df_tight = pl.DataFrame(
            {"y": [10.0], "q_0.1": [9.0], "q_0.5": [10.0], "q_0.9": [11.0]}
        )
        df_wide = pl.DataFrame(
            {"y": [10.0], "q_0.1": [5.0], "q_0.5": [10.0], "q_0.9": [15.0]}
        )
        assert crps(df_wide) > crps(df_tight)

    def test_biased_forecast_higher_crps(self):
        df_centered = pl.DataFrame(
            {"y": [10.0], "q_0.1": [8.0], "q_0.5": [10.0], "q_0.9": [12.0]}
        )
        df_biased = pl.DataFrame(
            {"y": [10.0], "q_0.1": [18.0], "q_0.5": [20.0], "q_0.9": [22.0]}
        )
        assert crps(df_biased) > crps(df_centered)

    def test_per_group(self):
        df = pl.DataFrame(
            {
                "uid": ["A", "A", "B", "B"],
                "y": [10.0, 20.0, 10.0, 20.0],
                "q_0.5": [10.0, 20.0, 15.0, 25.0],
            }
        )
        result = crps(df, id_col="uid")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2

    def test_no_quantile_cols_raises(self):
        df = pl.DataFrame({"y": [1.0], "pred": [1.0]})
        with pytest.raises(ValueError, match="No quantile columns"):
            crps(df)

    def test_length_mismatch_raises(self):
        df = pl.DataFrame({"y": [1.0], "q_0.5": [1.0]})
        with pytest.raises(ValueError, match="Length mismatch"):
            crps(df, quantile_cols=["q_0.5"], quantiles=[0.1, 0.5])

    def test_non_negative(self):
        df = pl.DataFrame(
            {"y": [10.0, 20.0], "q_0.5": [12.0, 18.0]}
        )
        assert crps(df) >= 0.0


# --- Namespace integration ---


class TestMetricsNamespace:
    def test_mae_via_namespace(self, simple_forecast):
        result = simple_forecast.pts.mae()
        assert result == 0.5

    def test_rmse_via_namespace(self, simple_forecast):
        result = simple_forecast.pts.rmse()
        assert result == 0.5

    def test_mape_via_namespace(self):
        df = pl.DataFrame({"y": [100.0, 200.0], "y_hat": [110.0, 180.0]})
        assert abs(df.pts.mape() - 0.1) < 1e-10

    def test_smape_via_namespace(self, perfect_forecast):
        assert perfect_forecast.pts.smape() == 0.0

    def test_crps_via_namespace(self):
        df = pl.DataFrame(
            {"y": [10.0], "q_0.1": [8.0], "q_0.5": [10.0], "q_0.9": [12.0]}
        )
        assert df.pts.crps() > 0.0


# --- Top-level import ---


class TestTopLevelImport:
    def test_import_mae(self):
        from polars_ts import mae as _mae

        assert callable(_mae)

    def test_import_rmse(self):
        from polars_ts import rmse as _rmse

        assert callable(_rmse)

    def test_import_smape(self):
        from polars_ts import smape as _smape

        assert callable(_smape)

    def test_import_crps(self):
        from polars_ts import crps as _crps

        assert callable(_crps)
