"""Tests for outlier detection and treatment (#61)."""

from datetime import date

import polars as pl
import pytest

from polars_ts.outliers import detect_outliers, treat_outliers


def _make_df_with_outlier() -> pl.DataFrame:
    values = [10.0, 11.0, 10.5, 10.2, 100.0, 10.3, 10.1, 10.4, 10.6, 10.0]
    return pl.DataFrame(
        {
            "unique_id": ["A"] * len(values),
            "ds": [date(2024, 1, i + 1) for i in range(len(values))],
            "y": values,
        }
    )


class TestDetectOutliers:
    def test_zscore(self):
        result = detect_outliers(_make_df_with_outlier(), method="zscore", threshold=2.0)
        assert "is_outlier" in result.columns
        outliers = result.filter(pl.col("is_outlier"))
        assert len(outliers) >= 1  # 100.0 should be detected

    def test_iqr(self):
        result = detect_outliers(_make_df_with_outlier(), method="iqr", threshold=1.5)
        outliers = result.filter(pl.col("is_outlier"))
        assert len(outliers) >= 1

    def test_hampel(self):
        result = detect_outliers(_make_df_with_outlier(), method="hampel", window=5)
        assert "is_outlier" in result.columns

    def test_rolling_zscore(self):
        result = detect_outliers(_make_df_with_outlier(), method="rolling_zscore", window=5, threshold=2.0)
        assert "is_outlier" in result.columns

    def test_rolling_requires_window(self):
        with pytest.raises(ValueError, match="window"):
            detect_outliers(_make_df_with_outlier(), method="rolling_zscore")

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            detect_outliers(_make_df_with_outlier(), method="invalid")

    def test_no_outliers(self):
        df = pl.DataFrame(
            {"unique_id": ["A"] * 5, "ds": [date(2024, 1, i + 1) for i in range(5)], "y": [1.0, 1.0, 1.0, 1.0, 1.0]}
        )
        result = detect_outliers(df, method="zscore")
        assert result.filter(pl.col("is_outlier")).height == 0


class TestTreatOutliers:
    def test_clip(self):
        result = treat_outliers(_make_df_with_outlier(), replacement="clip", threshold=2.0)
        assert "is_outlier" not in result.columns
        assert result["y"].max() < 100.0  # Outlier was clipped

    def test_null(self):
        result = treat_outliers(_make_df_with_outlier(), replacement="null", threshold=2.0)
        assert result["y"].null_count() >= 1

    def test_median(self):
        result = treat_outliers(_make_df_with_outlier(), replacement="median", threshold=2.0)
        assert result["y"].max() < 100.0

    def test_interpolate(self):
        result = treat_outliers(_make_df_with_outlier(), replacement="interpolate", threshold=2.0)
        assert result["y"].null_count() == 0
        assert result["y"].max() < 100.0

    def test_unknown_replacement(self):
        with pytest.raises(ValueError, match="Unknown replacement"):
            treat_outliers(_make_df_with_outlier(), replacement="invalid")


def test_constant_series_no_outliers():
    """Constant series should have zero variance and no outliers."""
    df = pl.DataFrame(
        {"unique_id": ["A"] * 10, "ds": [date(2024, 1, i + 1) for i in range(10)], "y": [5.0] * 10}
    )
    result = detect_outliers(df, method="zscore")
    assert result.filter(pl.col("is_outlier")).height == 0


def test_all_null_series():
    """Series with all nulls should not crash."""
    df = pl.DataFrame(
        {"unique_id": ["A"] * 5, "ds": [date(2024, 1, i + 1) for i in range(5)], "y": [None] * 5}
    ).cast({"y": pl.Float64})
    # Should not raise — nulls produce NaN z-scores which become False
    result = detect_outliers(df, method="zscore")
    assert "is_outlier" in result.columns


def test_per_group_independence():
    """Outlier detection should be computed independently per group."""
    # Group A: many 10s with one extreme 200 → outlier in A
    # Group B: values around 200 → 200 is NOT an outlier in B
    n = 20
    a_vals = [10.0] * (n - 1) + [200.0]
    b_vals = [199.0, 200.0, 201.0, 200.0, 200.0] * (n // 5)
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": a_vals + b_vals,
        }
    )
    result = detect_outliers(df, method="zscore", threshold=2.0)
    a_outliers = result.filter((pl.col("unique_id") == "A") & pl.col("is_outlier"))
    b_outliers = result.filter((pl.col("unique_id") == "B") & pl.col("is_outlier"))
    assert len(a_outliers) >= 1  # 200 is outlier in A
    assert len(b_outliers) == 0  # 200 is normal in B


def test_detect_treat_roundtrip():
    """Values flagged as outliers should be the ones replaced by treat."""
    df = _make_df_with_outlier()
    detected = detect_outliers(df, method="zscore", threshold=2.0)
    treated = treat_outliers(df, replacement="null", threshold=2.0)
    # Positions that were outliers should now be null
    outlier_mask = detected["is_outlier"].to_list()
    treated_nulls = treated["y"].is_null().to_list()
    for i, is_out in enumerate(outlier_mask):
        if is_out:
            assert treated_nulls[i], f"Outlier at index {i} was not nulled"


def test_rolling_zscore_window_larger_than_series():
    """Window larger than series should not crash."""
    df = pl.DataFrame(
        {"unique_id": ["A"] * 5, "ds": [date(2024, 1, i + 1) for i in range(5)], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    result = detect_outliers(df, method="rolling_zscore", window=100, threshold=2.0)
    assert "is_outlier" in result.columns


def test_hampel_varying_windows():
    """Hampel filter should work with different window sizes."""
    df = _make_df_with_outlier()
    for w in [3, 5, 7]:
        result = detect_outliers(df, method="hampel", window=w)
        assert "is_outlier" in result.columns


def test_boundary_value_zscore():
    """Value exactly at the threshold boundary."""
    # Construct data where we know the z-score of a value
    # mean=0, std=1 → value at exactly threshold=3.0 should NOT be outlier (> not >=)
    values = [0.0] * 99 + [3.0]
    df = pl.DataFrame(
        {"unique_id": ["A"] * 100, "ds": [date(2024, 1, 1)] * 100, "y": values}
    )
    result = detect_outliers(df, method="zscore", threshold=50.0)
    # With threshold=50, nothing should be an outlier
    assert result.filter(pl.col("is_outlier")).height == 0


def test_top_level_imports():
    import polars_ts

    assert polars_ts.detect_outliers is detect_outliers
    assert polars_ts.treat_outliers is treat_outliers
