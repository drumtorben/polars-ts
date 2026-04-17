"""Tests for Box-Cox transform and inverse."""

import math
from datetime import date

import polars as pl
import pytest

from polars_ts.transforms.boxcox import boxcox_transform, inverse_boxcox_transform


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "ds": [date(2024, 1, i) for i in range(1, 5)] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )


def test_boxcox_lambda_zero_is_log():
    df = _make_df()
    result = boxcox_transform(df, lam=0)

    expected = [math.log(v) for v in [1.0, 2.0, 3.0, 4.0]]
    actual = result.filter(pl.col("unique_id") == "A")["y"].to_list()
    for a, e in zip(actual, expected, strict=False):
        assert abs(a - e) < 1e-10


def test_boxcox_lambda_one_is_linear():
    df = _make_df()
    result = boxcox_transform(df, lam=1)

    # (y^1 - 1) / 1 = y - 1
    expected = [0.0, 1.0, 2.0, 3.0]
    actual = result.filter(pl.col("unique_id") == "A")["y"].to_list()
    for a, e in zip(actual, expected, strict=False):
        assert abs(a - e) < 1e-10


def test_boxcox_lambda_half():
    df = _make_df()
    result = boxcox_transform(df, lam=0.5)

    # (y^0.5 - 1) / 0.5 = 2*(sqrt(y) - 1)
    expected = [2 * (v**0.5 - 1) for v in [1.0, 2.0, 3.0, 4.0]]
    actual = result.filter(pl.col("unique_id") == "A")["y"].to_list()
    for a, e in zip(actual, expected, strict=False):
        assert abs(a - e) < 1e-10


def test_boxcox_roundtrip():
    df = _make_df()
    for lam in [0, 0.5, 1, 2, -0.5]:
        transformed = boxcox_transform(df, lam=lam)
        restored = inverse_boxcox_transform(transformed, lam=lam)

        for orig, rest in zip(df["y"].to_list(), restored["y"].to_list(), strict=False):
            assert abs(orig - rest) < 1e-8, f"Roundtrip failed for lambda={lam}"


def test_boxcox_roundtrip_lambda_from_column():
    df = _make_df()
    transformed = boxcox_transform(df, lam=0.5)
    restored = inverse_boxcox_transform(transformed)  # lam=None, reads from column

    for orig, rest in zip(df["y"].to_list(), restored["y"].to_list(), strict=False):
        assert abs(orig - rest) < 1e-8


def test_boxcox_lambda_column_stored():
    df = _make_df()
    result = boxcox_transform(df, lam=0.5)
    assert "y_boxcox_lambda" in result.columns
    assert result["y_boxcox_lambda"][0] == 0.5


def test_boxcox_original_column_created():
    df = _make_df()
    result = boxcox_transform(df, lam=1)
    assert "y_original" in result.columns
    assert result["y_original"].to_list() == df["y"].to_list()


def test_boxcox_inverse_drops_metadata():
    df = _make_df()
    transformed = boxcox_transform(df, lam=1)
    restored = inverse_boxcox_transform(transformed)
    assert "y_original" not in restored.columns
    assert "y_boxcox_lambda" not in restored.columns


def test_boxcox_non_positive_raises():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 3,
            "ds": [date(2024, 1, i) for i in range(1, 4)],
            "y": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        boxcox_transform(df, lam=1)


def test_boxcox_double_transform_raises():
    df = _make_df()
    transformed = boxcox_transform(df, lam=1)
    with pytest.raises(ValueError, match="already exists"):
        boxcox_transform(transformed, lam=1)


def test_boxcox_inverse_no_lambda_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="lam not provided"):
        inverse_boxcox_transform(df)


def test_boxcox_empty_dataframe():
    df = pl.DataFrame(
        {
            "unique_id": pl.Series([], dtype=pl.Utf8),
            "ds": pl.Series([], dtype=pl.Date),
            "y": pl.Series([], dtype=pl.Float64),
        }
    )
    result = boxcox_transform(df, lam=1)
    assert "y_original" in result.columns
    assert len(result) == 0


def test_boxcox_via_namespace():
    from polars_ts.metrics import Metrics  # noqa: F401

    df = _make_df()
    result = df.pts.boxcox_transform(lam=0.5)
    assert "y_boxcox_lambda" in result.columns
