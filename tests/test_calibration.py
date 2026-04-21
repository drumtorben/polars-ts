"""Tests for forecast calibration diagnostics (#58)."""

import polars as pl
import pytest

from polars_ts.calibration import calibration_table, pit_histogram, reliability_diagram


def _make_quantile_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "q_0.1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            "q_0.5": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "q_0.9": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        }
    )


class TestCalibrationTable:
    def test_basic(self):
        result = calibration_table(_make_quantile_df())
        assert "quantile" in result.columns
        assert "expected_coverage" in result.columns
        assert "observed_coverage" in result.columns
        assert len(result) == 3

    def test_coverage_increases_with_quantile(self):
        result = calibration_table(_make_quantile_df())
        coverages = result.sort("quantile")["observed_coverage"].to_list()
        # Higher quantiles should have higher or equal observed coverage
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1] - 0.01

    def test_no_quantile_cols_error(self):
        df = pl.DataFrame({"y": [1.0], "other": [2.0]})
        with pytest.raises(ValueError, match="quantile columns"):
            calibration_table(df)


class TestPITHistogram:
    def test_basic(self):
        result = pit_histogram(_make_quantile_df(), n_bins=5)
        assert "bin_lower" in result.columns
        assert "density" in result.columns
        assert len(result) == 5

    def test_density_sums_to_one(self):
        result = pit_histogram(_make_quantile_df(), n_bins=10)
        total = result["density"].sum()
        assert total == pytest.approx(1.0, abs=0.01)


class TestReliabilityDiagram:
    def test_basic(self):
        result = reliability_diagram(_make_quantile_df())
        assert result.columns == ["expected", "observed"]
        assert len(result) == 3


def test_calibration_well_calibrated():
    """Well-calibrated forecasts should have observed ≈ expected coverage."""
    df = _make_quantile_df()
    result = calibration_table(df)
    for row in result.iter_rows(named=True):
        assert abs(row["observed_coverage"] - row["expected_coverage"]) < 0.6


def test_pit_custom_bins():
    """PIT histogram should respect custom bin count."""
    for n_bins in [3, 5, 10]:
        result = pit_histogram(_make_quantile_df(), n_bins=n_bins)
        assert len(result) == n_bins


def test_reliability_diagram_sorted():
    """Reliability diagram should be sorted by expected."""
    result = reliability_diagram(_make_quantile_df())
    expected = result["expected"].to_list()
    assert expected == sorted(expected)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.calibration_table is calibration_table
    assert polars_ts.pit_histogram is pit_histogram
