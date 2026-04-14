import polars as pl
import pytest

from polars_ts import cusum


class TestCusumBasic:
    def test_constant_series(self):
        """CUSUM of a constant series should be all zeros."""
        df = pl.DataFrame({"unique_id": ["A"] * 5, "y": [3.0, 3.0, 3.0, 3.0, 3.0]})
        result = cusum(df)
        assert "cusum" in result.columns
        assert result["cusum"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_step_function(self):
        """CUSUM should show a clear V-shape at a mean shift."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 10,
                "y": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            }
        )
        result = cusum(df, normalize=False)
        vals = result["cusum"].to_list()
        # CUSUM decreases below zero in the first half (below mean)
        assert vals[0] < 0
        # Minimum occurs around the changepoint (index 4 or 5)
        min_idx = vals.index(min(vals))
        assert 3 <= min_idx <= 5
        # Sum of deviations from mean is zero by definition
        assert vals[-1] == pytest.approx(0.0)

    def test_unnormalized(self):
        """Unnormalized CUSUM should accumulate raw deviations."""
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        result = cusum(df, normalize=False)
        assert "cusum" in result.columns
        # Mean = 2.5; deviations = [-1.5, -0.5, 0.5, 1.5]; cumsum = [-1.5, -2.0, -1.5, 0.0]
        assert result["cusum"].to_list() == pytest.approx([-1.5, -2.0, -1.5, 0.0])

    def test_normalized(self):
        """Normalized CUSUM should divide by std before accumulating."""
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        result = cusum(df, normalize=True)
        assert "cusum" in result.columns
        # Should be unitless — values divided by std before cumsum
        vals = result["cusum"].to_list()
        assert vals[-1] == pytest.approx(0.0, abs=1e-10)


class TestCusumMultiGroup:
    def test_multiple_groups(self):
        """CUSUM should compute independently per group."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 30.0, 40.0],
            }
        )
        result = cusum(df, normalize=False)
        a_vals = result.filter(pl.col("unique_id") == "A")["cusum"].to_list()
        assert a_vals == pytest.approx([0.0, 0.0, 0.0, 0.0])


class TestCusumEdgeCases:
    def test_single_element_group(self):
        """Single-element group should produce cusum of 0."""
        df = pl.DataFrame({"unique_id": ["A"], "y": [5.0]})
        result = cusum(df, normalize=True)
        assert result["cusum"].to_list() == pytest.approx([0.0])

    def test_single_element_unnormalized(self):
        """Single-element group unnormalized should produce cusum of 0."""
        df = pl.DataFrame({"unique_id": ["A"], "y": [5.0]})
        result = cusum(df, normalize=False)
        assert result["cusum"].to_list() == pytest.approx([0.0])


class TestCusumErrors:
    def test_empty_dataframe(self):
        """Should raise ValueError on empty DataFrame."""
        df = pl.DataFrame({"unique_id": [], "y": []}).cast({"y": pl.Float64})
        with pytest.raises(ValueError, match="empty"):
            cusum(df)

    def test_missing_columns(self):
        """Should raise KeyError when required columns are missing."""
        df = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(KeyError, match="missing"):
            cusum(df)
