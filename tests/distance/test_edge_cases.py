"""Shared edge cases for all distance metrics."""

import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)

from tests.distance.conftest import _to_dict

UNIVARIATE_METRICS = [
    ("dtw", compute_pairwise_dtw, {}),
    ("ddtw", compute_pairwise_ddtw, {}),
    ("wdtw", compute_pairwise_wdtw, {}),
    ("msm", compute_pairwise_msm, {}),
    ("erp", compute_pairwise_erp, {}),
    ("lcss", compute_pairwise_lcss, {}),
    ("twe", compute_pairwise_twe, {}),
]


class TestVeryShortSeries:
    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_length_one(self, _name, fn, kwargs):
        """Series of length 1 should not panic."""
        df = pl.DataFrame(
            {
                "unique_id": ["A", "B"],
                "y": [1.0, 2.0],
            }
        )
        result = fn(df, df, **kwargs)
        assert result.shape[0] == 1

    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_length_two(self, _name, fn, kwargs):
        """Series of length 2 should work for all metrics."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 2 + ["B"] * 2,
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = fn(df, df, **kwargs)
        assert result.shape[0] == 1


class TestAllIdenticalValues:
    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_constant_series_identical(self, _name, fn, kwargs):
        """Two constant series with the same value should have distance 0 (or close)."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 5,
                "y": [3.0] * 5 + [3.0] * 5,
            }
        )
        result = fn(df, df, **kwargs)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_constant_series_different(self, _name, fn, kwargs):
        """Two constant series with different values should have positive distance."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 5,
                "y": [1.0] * 5 + [5.0] * 5,
            }
        )
        result = fn(df, df, **kwargs)
        d = _to_dict(result)
        # DDTW on constant series: derivative is all zeros, so distance = 0
        if _name == "ddtw":
            assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)
        else:
            assert d[("A", "B")] > 0


class TestExtremeValues:
    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_large_values(self, _name, fn, kwargs):
        """Very large values should not cause overflow or panic."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1e10, 2e10, 3e10, 4e10, 4e10, 3e10, 2e10, 1e10],
            }
        )
        result = fn(df, df, **kwargs)
        assert result.shape[0] == 1

    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_small_values(self, _name, fn, kwargs):
        """Very small values should work correctly."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1e-10, 2e-10, 3e-10, 4e-10, 4e-10, 3e-10, 2e-10, 1e-10],
            }
        )
        result = fn(df, df, **kwargs)
        assert result.shape[0] == 1


class TestUnequalLengthSeries:
    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_different_lengths(self, _name, fn, kwargs):
        """Series of different lengths should work."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 7,
                "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )
        result = fn(df, df, **kwargs)
        assert result.shape[0] == 1


class TestManySeries:
    @pytest.mark.parametrize("_name,fn,kwargs", UNIVARIATE_METRICS)
    def test_100_series_parallelism(self, _name, fn, kwargs):
        """100+ series to exercise parallel execution."""
        n_series = 100
        series_len = 10
        ids = []
        values = []
        for i in range(n_series):
            ids.extend([f"S{i}"] * series_len)
            values.extend([float(j + i * 0.1) for j in range(series_len)])

        df = pl.DataFrame({"unique_id": ids, "y": values})
        result = fn(df, df, **kwargs)
        # n*(n-1)/2 pairs
        expected_pairs = n_series * (n_series - 1) // 2
        assert result.shape[0] == expected_pairs
