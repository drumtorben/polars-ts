"""Tests for ForecastPipeline (Ch 8)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402

from polars_ts.pipeline import ForecastPipeline  # noqa: E402


def _make_ts(n: int = 30, n_series: int = 2) -> pl.DataFrame:
    """Create panel of linear-trend series with hourly timestamps."""
    rows: list[dict] = []
    base = datetime(2024, 1, 1)
    for s in range(n_series):
        sid = chr(65 + s)  # "A", "B", ...
        for i in range(n):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(hours=i),
                    "y": float(i * (s + 1)) + 10.0,  # different slopes
                }
            )
    return pl.DataFrame(rows)


class TestForecastPipeline:
    def test_fit_predict_lags_only(self):
        df = _make_ts()
        pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
        pipe.fit(df)
        result = pipe.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6  # 3 × 2 series

    def test_linear_trend_accuracy(self):
        df = _make_ts(n=30, n_series=1)
        pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
        pipe.fit(df)
        result = pipe.predict(df, h=1)

        pred = result["y_hat"][0]
        expected = 30.0 + 10.0  # slope=1, last=29+10=39, next≈40
        assert pred == pytest.approx(expected, abs=1.5)

    def test_rolling_features(self):
        df = _make_ts()
        pipe = ForecastPipeline(LinearRegression(), lags=[1], rolling_windows=[3])
        pipe.fit(df)
        result = pipe.predict(df, h=2)

        assert len(result) == 4

    def test_calendar_features(self):
        df = _make_ts()
        pipe = ForecastPipeline(Ridge(), lags=[1], calendar=["hour", "day_of_week"])
        pipe.fit(df)
        result = pipe.predict(df, h=2)

        assert len(result) == 4

    def test_fourier_features(self):
        df = _make_ts()
        pipe = ForecastPipeline(LinearRegression(), lags=[1], fourier=[(24, 2)])
        pipe.fit(df)
        result = pipe.predict(df, h=2)

        assert len(result) == 4

    def test_log_transform(self):
        df = _make_ts()
        pipe = ForecastPipeline(LinearRegression(), lags=[1, 2], target_transform="log")
        pipe.fit(df)
        result = pipe.predict(df, h=1)

        # Predictions should be on original scale (positive, similar magnitude)
        for v in result["y_hat"].to_list():
            assert v > 0

    def test_boxcox_transform(self):
        df = _make_ts()
        pipe = ForecastPipeline(
            LinearRegression(), lags=[1, 2], target_transform="boxcox", transform_kwargs={"lam": 0.5}
        )
        pipe.fit(df)
        result = pipe.predict(df, h=1)

        for v in result["y_hat"].to_list():
            assert v > 0

    def test_difference_transform(self):
        df = _make_ts(n=30, n_series=1)
        pipe = ForecastPipeline(LinearRegression(), lags=[1], target_transform="difference")
        pipe.fit(df)
        result = pipe.predict(df, h=2)

        # Predictions should be on original scale
        assert result["y_hat"][0] == pytest.approx(40.0, abs=2.0)

    def test_multiple_series(self):
        df = _make_ts(n_series=3)
        pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
        pipe.fit(df)
        result = pipe.predict(df, h=2)

        assert len(result) == 6  # 2 × 3 series
        ids = result["unique_id"].unique().to_list()
        assert len(ids) == 3

    def test_predict_before_fit(self):
        pipe = ForecastPipeline(LinearRegression(), lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            pipe.predict(_make_ts(), h=1)

    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="feature source"):
            ForecastPipeline(LinearRegression())

    def test_all_features_combined(self):
        df = _make_ts()
        pipe = ForecastPipeline(
            Ridge(),
            lags=[1, 2],
            rolling_windows=[3],
            calendar=["hour"],
            fourier=[(24, 1)],
        )
        pipe.fit(df)
        result = pipe.predict(df, h=1)

        assert len(result) == 2


def test_zero_horizon_raises():
    """Horizon of 0 should raise ValueError."""
    df = _make_ts()
    pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
    pipe.fit(df)
    with pytest.raises(ValueError, match="positive"):
        pipe.predict(df, h=0)


def test_negative_horizon_raises():
    """Negative horizon should raise ValueError."""
    df = _make_ts()
    pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
    pipe.fit(df)
    with pytest.raises(ValueError, match="positive"):
        pipe.predict(df, h=-1)


def test_single_series():
    """Pipeline should work with a single series."""
    df = _make_ts(n=30, n_series=1)
    pipe = ForecastPipeline(LinearRegression(), lags=[1, 2])
    pipe.fit(df)
    result = pipe.predict(df, h=3)
    assert len(result) == 3
    assert result["unique_id"].unique().to_list() == ["A"]


def test_predict_before_fit_message():
    """Error message should mention fit()."""
    pipe = ForecastPipeline(LinearRegression(), lags=[1])
    with pytest.raises(RuntimeError, match="fit"):
        pipe.predict(_make_ts(), h=1)


def test_feature_count_lags():
    """Number of feature columns should match number of lags."""
    df = _make_ts()
    pipe = ForecastPipeline(LinearRegression(), lags=[1, 2, 3])
    pipe.fit(df)
    assert len(pipe.feature_columns_) == 3


def test_feature_count_rolling():
    """Rolling features should produce window × aggs columns."""
    df = _make_ts()
    pipe = ForecastPipeline(LinearRegression(), lags=[1], rolling_windows=[3, 5])
    pipe.fit(df)
    # Default aggs: mean, std, min, max → 4 per window → 8 rolling + 1 lag = 9
    assert len(pipe.feature_columns_) == 9


def test_log_transform_roundtrip():
    """Log-transformed predictions should be on original scale and positive."""
    df = _make_ts(n=50, n_series=1)
    pipe = ForecastPipeline(Ridge(), lags=[1, 2], target_transform="log")
    pipe.fit(df)
    result = pipe.predict(df, h=3)
    for v in result["y_hat"].to_list():
        assert v > 0
    # Values should be in a reasonable range (original data is 10-60 ish)
    for v in result["y_hat"].to_list():
        assert 0 < v < 200


def test_top_level_import():
    import polars_ts

    assert polars_ts.ForecastPipeline is ForecastPipeline
