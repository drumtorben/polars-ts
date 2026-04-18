"""Tests for rolling window feature generation."""

from datetime import date

import polars as pl

from polars_ts.features.rolling import rolling_features


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": [date(2024, 1, i) for i in range(1, 7)] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_default_aggs():
    df = _make_df()
    result = rolling_features(df, windows=[3])

    for agg in ("mean", "std", "min", "max"):
        assert f"y_rolling_{agg}_3" in result.columns


def test_custom_aggs():
    df = _make_df()
    result = rolling_features(df, windows=[3], aggs=["sum", "median"])

    assert "y_rolling_sum_3" in result.columns
    assert "y_rolling_median_3" in result.columns
    assert "y_rolling_mean_3" not in result.columns


def test_multiple_windows():
    df = _make_df()
    result = rolling_features(df, windows=[2, 4], aggs=["mean"])

    assert "y_rolling_mean_2" in result.columns
    assert "y_rolling_mean_4" in result.columns


def test_rolling_mean_values():
    df = _make_df()
    result = rolling_features(df, windows=[3], aggs=["mean"])

    a_vals = result.filter(pl.col("unique_id") == "A")["y_rolling_mean_3"].to_list()
    # First two should be null (window=3, min_periods=3)
    assert a_vals[0] is None
    assert a_vals[1] is None
    assert abs(a_vals[2] - 2.0) < 1e-9  # mean(1,2,3)


def test_groups_independent():
    df = _make_df()
    result = rolling_features(df, windows=[3], aggs=["mean"])

    b_vals = result.filter(pl.col("unique_id") == "B")["y_rolling_mean_3"].to_list()
    assert b_vals[0] is None
    assert b_vals[1] is None
    assert abs(b_vals[2] - 20.0) < 1e-9  # mean(10,20,30)


def test_invalid_agg_raises():
    df = _make_df()
    import pytest

    with pytest.raises(ValueError):
        rolling_features(df, windows=[3], aggs=["bad"])


def test_min_periods():
    df = _make_df()
    result = rolling_features(df, windows=[3], aggs=["mean"], min_samples=1)

    a_vals = result.filter(pl.col("unique_id") == "A")["y_rolling_mean_3"].to_list()
    # With min_periods=1, first value should not be null
    assert a_vals[0] is not None


def test_preserves_row_count():
    df = _make_df()
    result = rolling_features(df, windows=[3])
    assert len(result) == len(df)


def test_negative_window_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        rolling_features(df, windows=[-1], aggs=["mean"])


def test_zero_window_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        rolling_features(df, windows=[0], aggs=["mean"])


def test_empty_dataframe():
    df = pl.DataFrame(
        {
            "unique_id": pl.Series([], dtype=pl.Utf8),
            "ds": pl.Series([], dtype=pl.Date),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    result = rolling_features(df, windows=[3], aggs=["mean"])
    assert "y_rolling_mean_3" in result.columns
    assert len(result) == 0


def test_rolling_via_namespace():
    """Test access through the pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401  — registers .pts namespace

    df = _make_df()
    result = df.pts.rolling_features(windows=[3], aggs=["mean"])
    assert "y_rolling_mean_3" in result.columns
