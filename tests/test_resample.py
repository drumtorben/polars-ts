"""Tests for temporal resampling (#62)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from polars_ts.resampling import resample


def _make_hourly_df() -> pl.DataFrame:
    n = 48
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)] * 2,
            "y": [float(i) for i in range(n)] + [float(2 * i) for i in range(n)],
        }
    )


class TestResample:
    def test_downsample_mean(self):
        result = resample(_make_hourly_df(), rule="1d", agg="mean")
        # 48 hours → 2 days per series
        a = result.filter(pl.col("unique_id") == "A")
        assert len(a) == 2

    def test_downsample_sum(self):
        result = resample(_make_hourly_df(), rule="1d", agg="sum")
        a = result.filter(pl.col("unique_id") == "A")
        assert len(a) == 2
        # First day sum = 0+1+...+23 = 276
        assert a["y"][0] == pytest.approx(276.0)

    def test_downsample_last(self):
        result = resample(_make_hourly_df(), rule="1d", agg="last")
        a = result.filter(pl.col("unique_id") == "A")
        assert a["y"][0] == pytest.approx(23.0)

    def test_multiple_series(self):
        result = resample(_make_hourly_df(), rule="1d", agg="mean")
        ids = result["unique_id"].unique().to_list()
        assert len(ids) == 2

    def test_unknown_agg(self):
        with pytest.raises(ValueError, match="Unknown agg"):
            resample(_make_hourly_df(), rule="1d", agg="invalid")

    def test_with_forward_fill(self):
        result = resample(_make_hourly_df(), rule="1d", agg="mean", fill="forward_fill")
        assert result["y"].null_count() == 0

    def test_group_independence(self):
        result = resample(_make_hourly_df(), rule="1d", agg="mean")
        a = result.filter(pl.col("unique_id") == "A")["y"][0]
        b = result.filter(pl.col("unique_id") == "B")["y"][0]
        assert b == pytest.approx(a * 2, abs=0.5)


def test_top_level_import():
    import polars_ts

    assert polars_ts.resample is resample
