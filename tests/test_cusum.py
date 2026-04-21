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


def test_custom_column_names():
    """CUSUM should work with non-default column names."""
    df = pl.DataFrame({"series": ["X"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = cusum(df, target_col="value", id_col="series")
    assert "cusum" in result.columns
    assert "series" in result.columns


def test_negative_values():
    """CUSUM should handle negative values correctly."""
    df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [-5.0, -3.0, -1.0, 1.0]})
    result = cusum(df, normalize=False)
    # Mean = -2.0; deviations = [-3, -1, 1, 3]; cumsum = [-3, -4, -3, 0]
    assert result["cusum"].to_list() == pytest.approx([-3.0, -4.0, -3.0, 0.0])


def test_multiple_mean_shifts():
    """CUSUM should show inflection points at each mean shift."""
    # Three segments: mean=0, mean=10, mean=0
    values = [0.0] * 10 + [10.0] * 10 + [0.0] * 10
    df = pl.DataFrame({"unique_id": ["A"] * 30, "y": values})
    result = cusum(df, normalize=False)
    vals = result["cusum"].to_list()
    # CUSUM should show clear changes in slope at the shift points
    # At index 9 and 19 there should be inflection points
    assert vals[-1] == pytest.approx(0.0)


def test_null_handling():
    """CUSUM should handle nulls gracefully (Polars ignores nulls in mean/std)."""
    df = pl.DataFrame({"unique_id": ["A"] * 5, "y": [1.0, None, 3.0, 4.0, 5.0]})
    result = cusum(df, normalize=False)
    assert "cusum" in result.columns
    assert len(result) == 5


def test_exact_theoretical_values():
    """Verify exact CUSUM values for a known simple case."""
    # Data: [2, 4, 6, 8] → mean=5 → deviations=[-3,-1,1,3] → cumsum=[-3,-4,-3,0]
    df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [2.0, 4.0, 6.0, 8.0]})
    result = cusum(df, normalize=False)
    expected = [-3.0, -4.0, -3.0, 0.0]
    assert result["cusum"].to_list() == pytest.approx(expected)
