"""Tests for Fourier feature generation."""

import math
from datetime import date

import polars as pl

from polars_ts.features.fourier import fourier_features


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 7 + ["B"] * 7,
            "ds": [date(2024, 1, i) for i in range(1, 8)] * 2,
            "y": list(range(7)) + list(range(10, 17)),
        }
    )


def test_single_harmonic():
    df = _make_df()
    result = fourier_features(df, period=7.0, n_harmonics=1)

    assert "fourier_sin_7.0_1" in result.columns
    assert "fourier_cos_7.0_1" in result.columns


def test_multiple_harmonics():
    df = _make_df()
    result = fourier_features(df, period=7.0, n_harmonics=3)

    for k in range(1, 4):
        assert f"fourier_sin_7.0_{k}" in result.columns
        assert f"fourier_cos_7.0_{k}" in result.columns
    assert len(result.columns) == len(df.columns) + 6


def test_sin_cos_values():
    df = _make_df()
    result = fourier_features(df, period=7.0, n_harmonics=1)

    a_sin = result.filter(pl.col("unique_id") == "A")["fourier_sin_7.0_1"].to_list()
    a_cos = result.filter(pl.col("unique_id") == "A")["fourier_cos_7.0_1"].to_list()

    # t=0 => sin(0)=0, cos(0)=1
    assert abs(a_sin[0] - 0.0) < 1e-9
    assert abs(a_cos[0] - 1.0) < 1e-9

    # t=1 => sin(2*pi/7), cos(2*pi/7)
    expected_sin = math.sin(2 * math.pi / 7)
    expected_cos = math.cos(2 * math.pi / 7)
    assert abs(a_sin[1] - expected_sin) < 1e-9
    assert abs(a_cos[1] - expected_cos) < 1e-9


def test_groups_independent():
    """Each group should have its own 0-based time index."""
    df = _make_df()
    result = fourier_features(df, period=7.0, n_harmonics=1)

    b_sin = result.filter(pl.col("unique_id") == "B")["fourier_sin_7.0_1"].to_list()
    # First element of group B should also be sin(0) = 0
    assert abs(b_sin[0] - 0.0) < 1e-9


def test_period_completes_cycle():
    """After exactly one period, sin/cos should return to starting values."""
    # Check that sin(2*pi*7/7) ≈ sin(0) ≈ 0
    last_plus_one_sin = math.sin(2 * math.pi * 7 / 7)
    assert abs(last_plus_one_sin - 0.0) < 1e-9


def test_preserves_row_count():
    df = _make_df()
    result = fourier_features(df, period=7.0, n_harmonics=2)
    assert len(result) == len(df)


def test_zero_period_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        fourier_features(df, period=0)


def test_negative_period_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="positive"):
        fourier_features(df, period=-7.0)


def test_zero_harmonics_raises():
    import pytest

    df = _make_df()
    with pytest.raises(ValueError, match="n_harmonics"):
        fourier_features(df, period=7.0, n_harmonics=0)


def test_integer_period_normalizes_to_float():
    df = _make_df()
    result = fourier_features(df, period=7, n_harmonics=1)
    # Should use float in column name regardless of input type
    assert "fourier_sin_7.0_1" in result.columns


def test_fourier_via_namespace():
    """Test access through the pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401  — registers .pts namespace

    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 7,
            "ds": [date(2024, 1, i) for i in range(1, 8)],
            "y": list(range(7)),
        }
    )
    result = df.pts.fourier_features(period=7.0, n_harmonics=1)
    assert "fourier_sin_7.0_1" in result.columns
