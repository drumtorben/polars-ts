"""Tests for calendar feature extraction."""

from datetime import datetime

import polars as pl

from polars_ts.features.calendar import calendar_features


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ds": [
                datetime(2024, 1, 1, 10, 30),  # Monday
                datetime(2024, 1, 6, 14, 0),  # Saturday
                datetime(2024, 3, 15, 8, 15),  # Friday
                datetime(2024, 12, 31, 23, 59),  # Tuesday
            ],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_default_features():
    df = _make_df()
    result = calendar_features(df)

    expected = [
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week",
        "month",
        "quarter",
        "year",
        "hour",
        "minute",
        "is_weekend",
    ]
    for feat in expected:
        assert feat in result.columns


def test_selected_features():
    df = _make_df()
    result = calendar_features(df, features=["month", "quarter"])

    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "day_of_week" not in result.columns


def test_month_values():
    df = _make_df()
    result = calendar_features(df, features=["month"])
    assert result["month"].to_list() == [1, 1, 3, 12]


def test_quarter_values():
    df = _make_df()
    result = calendar_features(df, features=["quarter"])
    assert result["quarter"].to_list() == [1, 1, 1, 4]


def test_is_weekend():
    df = _make_df()
    result = calendar_features(df, features=["is_weekend"])
    # Monday=0, Saturday=1, Friday=0, Tuesday=0
    # Polars weekday: Monday=1..Sunday=7, so >= 6 means Sat/Sun
    vals = result["is_weekend"].to_list()
    assert vals[0] == 0  # Monday
    assert vals[1] == 1  # Saturday
    assert vals[2] == 0  # Friday
    assert vals[3] == 0  # Tuesday


def test_hour_minute():
    df = _make_df()
    result = calendar_features(df, features=["hour", "minute"])
    assert result["hour"].to_list() == [10, 14, 8, 23]
    assert result["minute"].to_list() == [30, 0, 15, 59]


def test_invalid_feature_raises():
    df = _make_df()
    import pytest

    with pytest.raises(ValueError):
        calendar_features(df, features=["bad_feature"])


def test_preserves_original_columns():
    df = _make_df()
    result = calendar_features(df, features=["month"])
    assert "ds" in result.columns
    assert "y" in result.columns
    assert len(result) == len(df)


def test_calendar_via_namespace():
    """Test access through the pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401  — registers .pts namespace

    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 4,
            "ds": [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
                datetime(2024, 4, 1),
            ],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = df.pts.calendar_features(features=["month"])
    assert "month" in result.columns
