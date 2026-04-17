"""Tests for log transform and inverse."""

import math
from datetime import date

import polars as pl
import pytest

from polars_ts.transforms.log import inverse_log_transform, log_transform


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "ds": [date(2024, 1, i) for i in range(1, 5)] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


def test_log_transform_values():
    df = _make_df()
    result = log_transform(df)

    expected = [math.log1p(v) for v in [1.0, 2.0, 3.0, 4.0]]
    actual = result.filter(pl.col("unique_id") == "A")["y"].to_list()
    for a, e in zip(actual, expected, strict=False):
        assert abs(a - e) < 1e-10


def test_log_original_column_created():
    df = _make_df()
    result = log_transform(df)
    assert "y_original" in result.columns
    assert result["y_original"].to_list() == df["y"].to_list()


def test_log_roundtrip():
    df = _make_df()
    transformed = log_transform(df)
    restored = inverse_log_transform(transformed)

    for orig, rest in zip(df["y"].to_list(), restored["y"].to_list(), strict=False):
        assert abs(orig - rest) < 1e-10


def test_log_inverse_drops_metadata():
    df = _make_df()
    transformed = log_transform(df)
    restored = inverse_log_transform(transformed)
    assert "y_original" not in restored.columns


def test_log_zero_values():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": [date(2024, 1, i) for i in range(1, 4)],
            "y": [0.0, 1.0, 2.0],
        }
    )
    result = log_transform(df)
    assert abs(result["y"][0] - 0.0) < 1e-10  # log1p(0) = 0


def test_log_minus_one_raises():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": [date(2024, 1, i) for i in range(1, 4)],
            "y": [-1.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="log1p requires"):
        log_transform(df)


def test_log_negative_values_raises():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": [date(2024, 1, i) for i in range(1, 4)],
            "y": [-2.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="log1p requires"):
        log_transform(df)


def test_log_double_transform_raises():
    df = _make_df()
    transformed = log_transform(df)
    with pytest.raises(ValueError, match="already exists"):
        log_transform(transformed)


def test_log_preserves_columns():
    df = _make_df()
    result = log_transform(df)
    for col in df.columns:
        assert col in result.columns
    assert len(result) == len(df)


def test_log_empty_dataframe():
    df = pl.DataFrame(
        {
            "unique_id": pl.Series([], dtype=pl.Utf8),
            "ds": pl.Series([], dtype=pl.Date),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    result = log_transform(df)
    assert "y_original" in result.columns
    assert len(result) == 0


def test_log_via_namespace():
    from polars_ts.metrics import Metrics  # noqa: F401

    df = _make_df()
    result = df.pts.log_transform()
    assert "y_original" in result.columns
