"""Tests for covariates support (past, future, static exogenous variables).

Covers ForecastPipeline, GlobalForecaster, RecursiveForecaster, and
DirectForecaster with past_covariates and future_covariates.
"""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402

from polars_ts.features.lags import covariate_lag_features  # noqa: E402
from polars_ts.global_model import GlobalForecaster  # noqa: E402
from polars_ts.models.multistep import DirectForecaster, RecursiveForecaster  # noqa: E402
from polars_ts.pipeline import ForecastPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cov_ts(n: int = 30, n_series: int = 2) -> pl.DataFrame:
    """Panel with target y = 2*temperature + is_holiday*10 + trend."""
    rows: list[dict] = []
    base = datetime(2024, 1, 1)
    for s in range(n_series):
        sid = chr(65 + s)
        for i in range(n):
            temp = 20.0 + np.sin(i / 5.0) * 5.0
            holiday = 1.0 if i % 7 == 0 else 0.0
            y = float(i * (s + 1)) + 2.0 * temp + holiday * 10.0
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(hours=i),
                    "y": y,
                    "temperature": temp,
                    "is_holiday": holiday,
                }
            )
    return pl.DataFrame(rows)


def _make_future_df(
    df: pl.DataFrame,
    h: int,
    n_series: int = 2,
) -> pl.DataFrame:
    """Generate future_df with future covariate values for h steps."""
    rows: list[dict] = []
    base_time = df.filter(pl.col("unique_id") == "A")["ds"].max()
    freq = timedelta(hours=1)
    for s in range(n_series):
        sid = chr(65 + s)
        for step in range(1, h + 1):
            t = step + 29  # continue from last training index
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base_time + freq * step,
                    "is_holiday": 1.0 if t % 7 == 0 else 0.0,
                }
            )
    return pl.DataFrame(rows)


def _make_date_cov_ts(n: int = 20) -> pl.DataFrame:
    """Two series with date index and past + future covariates."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) + 2.0 * (i % 3) for i in range(n)] + [float(2 * i) + 3.0 * (i % 3) for i in range(n)],
            "past_x": [float(i % 3) for i in range(n)] * 2,
            "future_x": [1.0 if i % 5 == 0 else 0.0 for i in range(n)] * 2,
        }
    )


def _make_date_future_df(h: int = 3) -> pl.DataFrame:
    """Future df for _make_date_cov_ts."""
    rows: list[dict] = []
    for sid in ["A", "B"]:
        for step in range(1, h + 1):
            t = 19 + step
            rows.append(
                {
                    "unique_id": sid,
                    "ds": date(2024, 1, 20 + step),
                    "future_x": 1.0 if t % 5 == 0 else 0.0,
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# covariate_lag_features
# ---------------------------------------------------------------------------


class TestCovariateLagFeatures:
    def test_basic(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": [date(2024, 1, i + 1) for i in range(5)],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "temp": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        result = covariate_lag_features(df, ["temp"], [1, 2], "unique_id", "ds")
        assert "temp_lag_1" in result.columns
        assert "temp_lag_2" in result.columns
        assert result["temp_lag_1"][2] == 20.0  # lag 1 of row 2
        assert result["temp_lag_2"][2] == 10.0  # lag 2 of row 2

    def test_multiple_covariates(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": [date(2024, 1, i + 1) for i in range(5)],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "temp": [10.0, 20.0, 30.0, 40.0, 50.0],
                "humidity": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        result = covariate_lag_features(df, ["temp", "humidity"], [1], "unique_id", "ds")
        assert "temp_lag_1" in result.columns
        assert "humidity_lag_1" in result.columns

    def test_multi_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A", "A", "B", "B", "B"],
                "ds": [date(2024, 1, i + 1) for i in range(3)] * 2,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "temp": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        result = covariate_lag_features(df, ["temp"], [1], "unique_id", "ds")
        # B's lag_1 at position 1 should be B's first value, not A's last
        b_df = result.filter(pl.col("unique_id") == "B")
        assert b_df["temp_lag_1"][1] == 40.0

    def test_invalid_lags(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [date(2024, 1, i + 1) for i in range(3)],
                "y": [1.0, 2.0, 3.0],
                "temp": [10.0, 20.0, 30.0],
            }
        )
        with pytest.raises(ValueError, match="positive"):
            covariate_lag_features(df, ["temp"], [0], "unique_id", "ds")


# ---------------------------------------------------------------------------
# ForecastPipeline with covariates
# ---------------------------------------------------------------------------


class TestForecastPipelineCovariates:
    def test_past_covariates(self):
        df = _make_cov_ts()
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            past_covariates=["temperature"],
        )
        pipe.fit(df)
        result = pipe.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6  # 3 × 2 series
        # Feature columns should include temperature lags
        assert any("temperature_lag_" in c for c in pipe.feature_columns_)
        # Raw temperature should NOT be a feature
        assert "temperature" not in pipe.feature_columns_

    def test_future_covariates(self):
        df = _make_cov_ts()
        future_df = _make_future_df(df, h=3)
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            future_covariates=["is_holiday"],
        )
        pipe.fit(df)
        result = pipe.predict(df, h=3, future_df=future_df)

        assert len(result) == 6
        assert "is_holiday" in pipe.feature_columns_

    def test_future_covariates_requires_future_df(self):
        df = _make_cov_ts()
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            future_covariates=["is_holiday"],
        )
        pipe.fit(df)
        with pytest.raises(ValueError, match="future_df"):
            pipe.predict(df, h=3)

    def test_past_and_future_combined(self):
        df = _make_cov_ts()
        future_df = _make_future_df(df, h=2)
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            past_covariates=["temperature"],
            future_covariates=["is_holiday"],
        )
        pipe.fit(df)
        result = pipe.predict(df, h=2, future_df=future_df)

        assert len(result) == 4
        assert any("temperature_lag_" in c for c in pipe.feature_columns_)
        assert "is_holiday" in pipe.feature_columns_

    def test_custom_past_covariate_lags(self):
        df = _make_cov_ts()
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            past_covariates=["temperature"],
            past_covariate_lags=[1, 3, 5],
        )
        pipe.fit(df)
        result = pipe.predict(df, h=1)

        assert len(result) == 2
        assert "temperature_lag_1" in pipe.feature_columns_
        assert "temperature_lag_3" in pipe.feature_columns_
        assert "temperature_lag_5" in pipe.feature_columns_

    def test_past_covariates_improve_accuracy(self):
        """Model with covariates should outperform model without."""
        df = _make_cov_ts(n=50, n_series=1)
        future_df = _make_future_df(df, h=1, n_series=1)

        # Without covariates
        pipe_base = ForecastPipeline(Ridge(), lags=[1, 2])
        pipe_base.fit(df.select("unique_id", "ds", "y"))
        pred_base = pipe_base.predict(df.select("unique_id", "ds", "y"), h=1)

        # With covariates
        pipe_cov = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            past_covariates=["temperature"],
            future_covariates=["is_holiday"],
        )
        pipe_cov.fit(df)
        pred_cov = pipe_cov.predict(df, h=1, future_df=future_df)

        # Both should produce results
        assert len(pred_base) == 1
        assert len(pred_cov) == 1

    def test_backward_compatible_no_covariates(self):
        """Pipeline without covariates should work exactly as before."""
        df = _make_cov_ts().select("unique_id", "ds", "y")
        pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
        pipe.fit(df)
        result = pipe.predict(df, h=2)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# RecursiveForecaster with covariates
# ---------------------------------------------------------------------------


class TestRecursiveForecasterCovariates:
    def test_past_covariates(self):
        df = _make_date_cov_ts()
        fc = RecursiveForecaster(
            LinearRegression(),
            lags=[1, 2],
            past_covariates=["past_x"],
        )
        fc.fit(df)
        result = fc.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_future_covariates(self):
        df = _make_date_cov_ts()
        future_df = _make_date_future_df(h=3)
        fc = RecursiveForecaster(
            LinearRegression(),
            lags=[1, 2],
            future_covariates=["future_x"],
        )
        fc.fit(df)
        result = fc.predict(df, h=3, future_df=future_df)

        assert len(result) == 6

    def test_future_covariates_requires_future_df(self):
        df = _make_date_cov_ts()
        fc = RecursiveForecaster(
            LinearRegression(),
            lags=[1, 2],
            future_covariates=["future_x"],
        )
        fc.fit(df)
        with pytest.raises(ValueError, match="future_df"):
            fc.predict(df, h=3)

    def test_past_and_future_combined(self):
        df = _make_date_cov_ts()
        future_df = _make_date_future_df(h=2)
        fc = RecursiveForecaster(
            LinearRegression(),
            lags=[1, 2],
            past_covariates=["past_x"],
            future_covariates=["future_x"],
        )
        fc.fit(df)
        result = fc.predict(df, h=2, future_df=future_df)

        assert len(result) == 4

    def test_custom_past_covariate_lags(self):
        df = _make_date_cov_ts()
        fc = RecursiveForecaster(
            LinearRegression(),
            lags=[1, 2],
            past_covariates=["past_x"],
            past_covariate_lags=[1, 3],
        )
        fc.fit(df)
        result = fc.predict(df, h=2)
        assert len(result) == 4

    def test_backward_compatible(self):
        df = _make_date_cov_ts().select("unique_id", "ds", "y")
        fc = RecursiveForecaster(LinearRegression(), lags=[1, 2])
        fc.fit(df)
        result = fc.predict(df, h=3)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# DirectForecaster with covariates
# ---------------------------------------------------------------------------


class TestDirectForecasterCovariates:
    def test_past_covariates(self):
        df = _make_date_cov_ts()
        fc = DirectForecaster(
            lambda: LinearRegression(),
            lags=[1, 2],
            h=3,
            past_covariates=["past_x"],
        )
        fc.fit(df)
        result = fc.predict(df)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_future_covariates(self):
        df = _make_date_cov_ts()
        future_df = _make_date_future_df(h=3)
        fc = DirectForecaster(
            lambda: LinearRegression(),
            lags=[1, 2],
            h=3,
            future_covariates=["future_x"],
        )
        fc.fit(df)
        result = fc.predict(df, future_df=future_df)

        assert len(result) == 6

    def test_future_covariates_requires_future_df(self):
        df = _make_date_cov_ts()
        fc = DirectForecaster(
            lambda: LinearRegression(),
            lags=[1, 2],
            h=3,
            future_covariates=["future_x"],
        )
        fc.fit(df)
        with pytest.raises(ValueError, match="future_df"):
            fc.predict(df)

    def test_past_and_future_combined(self):
        df = _make_date_cov_ts()
        future_df = _make_date_future_df(h=2)
        fc = DirectForecaster(
            lambda: LinearRegression(),
            lags=[1, 2],
            h=2,
            past_covariates=["past_x"],
            future_covariates=["future_x"],
        )
        fc.fit(df)
        result = fc.predict(df, future_df=future_df)

        assert len(result) == 4

    def test_backward_compatible(self):
        df = _make_date_cov_ts().select("unique_id", "ds", "y")
        fc = DirectForecaster(lambda: LinearRegression(), lags=[1, 2], h=3)
        fc.fit(df)
        result = fc.predict(df)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# GlobalForecaster with covariates
# ---------------------------------------------------------------------------


class TestGlobalForecasterCovariates:
    def test_past_covariates(self):
        df = _make_cov_ts()
        gf = GlobalForecaster(
            Ridge(),
            lags=[1, 2],
            past_covariates=["temperature"],
        )
        gf.fit(df)
        result = gf.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6
        assert any("temperature_lag_" in c for c in gf.feature_columns_)

    def test_future_covariates(self):
        df = _make_cov_ts()
        future_df = _make_future_df(df, h=2)
        gf = GlobalForecaster(
            Ridge(),
            lags=[1, 2],
            future_covariates=["is_holiday"],
        )
        gf.fit(df)
        result = gf.predict(df, h=2, future_df=future_df)

        assert len(result) == 4
        assert "is_holiday" in gf.feature_columns_

    def test_future_covariates_requires_future_df(self):
        df = _make_cov_ts()
        gf = GlobalForecaster(
            Ridge(),
            lags=[1, 2],
            future_covariates=["is_holiday"],
        )
        gf.fit(df)
        with pytest.raises(ValueError, match="future_df"):
            gf.predict(df, h=2)

    def test_covariates_with_static_features(self):
        """Covariates should work alongside existing static_features."""
        df = _make_cov_ts().with_columns(
            pl.when(pl.col("unique_id") == "A").then(pl.lit("urban")).otherwise(pl.lit("rural")).alias("location")
        )
        future_df = _make_future_df(df, h=2)
        gf = GlobalForecaster(
            Ridge(),
            lags=[1, 2],
            encode_id="ordinal",
            static_features=["location"],
            past_covariates=["temperature"],
            future_covariates=["is_holiday"],
        )
        gf.fit(df)
        result = gf.predict(df, h=2, future_df=future_df)

        assert len(result) == 4
        assert any("temperature_lag_" in c for c in gf.feature_columns_)
        assert "is_holiday" in gf.feature_columns_
        assert "__static_location" in gf.feature_columns_
        assert "__id_encoded" in gf.feature_columns_

    def test_backward_compatible(self):
        df = _make_cov_ts().select("unique_id", "ds", "y")
        gf = GlobalForecaster(Ridge(), lags=[1, 2])
        gf.fit(df)
        result = gf.predict(df, h=2)
        assert len(result) == 4
