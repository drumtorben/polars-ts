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


def test_top_level_imports():
    import polars_ts

    assert polars_ts.detect_outliers is detect_outliers
    assert polars_ts.treat_outliers is treat_outliers
