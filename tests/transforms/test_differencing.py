"""Tests for differencing and undifferencing."""

from datetime import date

import polars as pl
import pytest

from polars_ts.transforms.differencing import difference, undifference


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": [date(2024, 1, i) for i in range(1, 7)] * 2,
            "y": [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_difference_first_order():
    df = _make_df()
    result = difference(df)

    a_vals = result.filter(pl.col("unique_id") == "A")["y"].to_list()
    # y = [1, 3, 6, 10, 15, 21] => diff = [2, 3, 4, 5, 6]
    assert len(a_vals) == 5
    for a, e in zip(a_vals, [2.0, 3.0, 4.0, 5.0, 6.0], strict=False):
        assert abs(a - e) < 1e-10


def test_difference_groups_independent():
    df = _make_df()
    result = difference(df)

    b_vals = result.filter(pl.col("unique_id") == "B")["y"].to_list()
    # y = [10, 20, 30, 40, 50, 60] => diff = [10, 10, 10, 10, 10]
    assert len(b_vals) == 5
    for v in b_vals:
        assert abs(v - 10.0) < 1e-10


def test_difference_seasonal():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": [date(2024, 1, i) for i in range(1, 9)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        }
    )
    result = difference(df, period=4)

    # y_t - y_{t-4}: [5-1, 7-2, 9-3, 11-4] = [4, 5, 6, 7]
    vals = result["y"].to_list()
    assert len(vals) == 4
    for a, e in zip(vals, [4.0, 5.0, 6.0, 7.0], strict=False):
        assert abs(a - e) < 1e-10


def test_difference_second_order():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 5,
            "ds": [date(2024, 1, i) for i in range(1, 6)],
            "y": [1.0, 3.0, 6.0, 10.0, 15.0],
        }
    )
    result = difference(df, order=2)

    # First diff: [2, 3, 4, 5], second diff: [1, 1, 1]
    vals = result["y"].to_list()
    assert len(vals) == 3
    for v in vals:
        assert abs(v - 1.0) < 1e-10


def test_difference_initial_values_stored():
    df = _make_df()
    result = difference(df)

    assert "y_diff_initial" in result.columns
    # For order=1, period=1, first 1 value per group is stored
    a_init = result.filter(pl.col("unique_id") == "A")["y_diff_initial"][0].to_list()
    assert a_init == [1.0]


def test_undifference_roundtrip():
    df = _make_df()
    diffed = difference(df)
    restored = undifference(diffed)

    # Compare restored values against originals (minus the first dropped row per group)
    orig_a = df.filter(pl.col("unique_id") == "A")["y"].to_list()[1:]
    rest_a = restored.filter(pl.col("unique_id") == "A")["y"].to_list()
    assert len(rest_a) == len(orig_a)
    for o, r in zip(orig_a, rest_a, strict=False):
        assert abs(o - r) < 1e-10


def test_undifference_seasonal_roundtrip():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": [date(2024, 1, i) for i in range(1, 9)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        }
    )
    diffed = difference(df, period=4)
    restored = undifference(diffed, period=4)

    orig = df["y"].to_list()[4:]  # first 4 dropped
    rest = restored["y"].to_list()
    assert len(rest) == len(orig)
    for o, r in zip(orig, rest, strict=False):
        assert abs(o - r) < 1e-10


def test_undifference_drops_metadata():
    df = _make_df()
    diffed = difference(df)
    restored = undifference(diffed)
    assert "y_diff_initial" not in restored.columns


def test_difference_invalid_order_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="order"):
        difference(df, order=0)


def test_difference_invalid_period_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="period"):
        difference(df, period=0)


def test_difference_double_raises():
    df = _make_df()
    diffed = difference(df)
    with pytest.raises(ValueError, match="already exists"):
        difference(diffed)


def test_difference_empty_dataframe():
    df = pl.DataFrame(
        {
            "unique_id": pl.Series([], dtype=pl.Utf8),
            "ds": pl.Series([], dtype=pl.Date),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    result = difference(df)
    assert "y_diff_initial" in result.columns
    assert len(result) == 0


def test_undifference_second_order_roundtrip():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 5,
            "ds": [date(2024, 1, i) for i in range(1, 6)],
            "y": [1.0, 3.0, 6.0, 10.0, 15.0],
        }
    )
    diffed = difference(df, order=2)
    restored = undifference(diffed, order=2)

    # First 2 rows dropped (order=2, period=1), remaining: [6, 10, 15]
    orig = df["y"].to_list()[2:]
    rest = restored["y"].to_list()
    assert len(rest) == len(orig)
    for o, r in zip(orig, rest, strict=False):
        assert abs(o - r) < 1e-10


def test_undifference_second_order_multigroup():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 5 + ["B"] * 5,
            "ds": [date(2024, 1, i) for i in range(1, 6)] * 2,
            "y": [1.0, 3.0, 6.0, 10.0, 15.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    diffed = difference(df, order=2)
    restored = undifference(diffed, order=2)

    for uid in ["A", "B"]:
        orig = df.filter(pl.col("unique_id") == uid)["y"].to_list()[2:]
        rest = restored.filter(pl.col("unique_id") == uid)["y"].to_list()
        assert len(rest) == len(orig), f"Length mismatch for group {uid}"
        for o, r in zip(orig, rest, strict=False):
            assert abs(o - r) < 1e-10, f"Value mismatch for group {uid}: {o} vs {r}"


def test_undifference_second_order_seasonal_roundtrip():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 12,
            "ds": [date(2024, 1, i) for i in range(1, 13)],
            "y": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        }
    )
    diffed = difference(df, order=2, period=4)
    restored = undifference(diffed, order=2, period=4)

    # order=2, period=4 drops first 8 rows
    orig = df["y"].to_list()[8:]
    rest = restored["y"].to_list()
    assert len(rest) == len(orig)
    for o, r in zip(orig, rest, strict=False):
        assert abs(o - r) < 1e-10


def test_difference_via_namespace():
    from polars_ts.metrics import Metrics  # noqa: F401

    df = _make_df()
    result = df.pts.difference()
    assert "y_diff_initial" in result.columns
