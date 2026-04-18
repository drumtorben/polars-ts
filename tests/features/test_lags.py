"""Tests for lag feature generation."""

from datetime import date

import polars as pl

from polars_ts.features.lags import lag_features


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "ds": [date(2024, 1, i) for i in range(1, 5)] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


def test_single_lag():
    df = _make_df()
    result = lag_features(df, lags=[1])

    assert "y_lag_1" in result.columns
    # First value in each group should be null
    a_lags = result.filter(pl.col("unique_id") == "A")["y_lag_1"].to_list()
    assert a_lags[0] is None
    assert a_lags[1:] == [1.0, 2.0, 3.0]


def test_multiple_lags():
    df = _make_df()
    result = lag_features(df, lags=[1, 2])

    assert "y_lag_1" in result.columns
    assert "y_lag_2" in result.columns


def test_groups_independent():
    df = _make_df()
    result = lag_features(df, lags=[1])

    b_lags = result.filter(pl.col("unique_id") == "B")["y_lag_1"].to_list()
    assert b_lags[0] is None
    assert b_lags[1:] == [10.0, 20.0, 30.0]


def test_custom_target_col():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "ds": [date(2024, 1, i) for i in range(1, 5)] * 2,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    result = lag_features(df, lags=[1], target_col="value")

    assert "value_lag_1" in result.columns


def test_preserves_all_columns():
    df = _make_df()
    result = lag_features(df, lags=[1, 3])

    for col in df.columns:
        assert col in result.columns
    assert len(result) == len(df)


def test_negative_lag_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        lag_features(df, lags=[-1])


def test_zero_lag_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        lag_features(df, lags=[0])


def test_empty_dataframe():
    df = pl.DataFrame(
        {
            "unique_id": pl.Series([], dtype=pl.Utf8),
            "ds": pl.Series([], dtype=pl.Date),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    result = lag_features(df, lags=[1])
    assert "y_lag_1" in result.columns
    assert len(result) == 0


def test_single_row_group():
    df = pl.DataFrame({"unique_id": ["A"], "ds": [date(2024, 1, 1)], "y": [1.0]})
    result = lag_features(df, lags=[1])
    assert result["y_lag_1"].to_list() == [None]


def test_unsorted_input():
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
            "y": [30.0, 10.0, 20.0],
        }
    )
    result = lag_features(df, lags=[1])
    # After sorting by ds: y=[10, 20, 30], lag_1=[None, 10, 20]
    assert result["y_lag_1"].to_list() == [None, 10.0, 20.0]


def test_lag_via_namespace():
    """Test access through the pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401  — registers .pts namespace

    df = _make_df()
    result = df.pts.lag_features(lags=[1])
    assert "y_lag_1" in result.columns
