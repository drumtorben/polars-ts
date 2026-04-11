import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import compute_pairwise_ddtw

from tests.distance.conftest import _to_dict


class TestDDTWBasic:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_ddtw(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance_positive(self, two_series):
        result = compute_pairwise_ddtw(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] > 0

    def test_output_columns(self, two_series):
        result = compute_pairwise_ddtw(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "ddtw"}

    def test_single_series_empty(self, single_series):
        result = compute_pairwise_ddtw(single_series, single_series)
        assert result.shape[0] == 0

    def test_three_series_pairs(self, three_series):
        result = compute_pairwise_ddtw(three_series, three_series)
        d = _to_dict(result)
        assert len(d) == 3

    def test_non_negativity(self, three_series):
        result = compute_pairwise_ddtw(three_series, three_series)
        assert (result["ddtw"] >= 0).all()

    def test_int_id_preserved(self, int_id_series):
        result = compute_pairwise_ddtw(int_id_series, int_id_series)
        assert result["id_1"].dtype == pl.Int64
        assert result["id_2"].dtype == pl.Int64


class TestDDTWProperties:
    def test_symmetry(self, three_series):
        """DDTW(A,C) computed from both directions should give the same result."""
        df_a = three_series.filter(pl.col("unique_id") == "A")
        df_c = three_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_ddtw(df_a, df_c)
        ca = compute_pairwise_ddtw(df_c, df_a)
        assert abs(ac["ddtw"][0] - ca["ddtw"][0]) < 1e-10

    def test_no_self_comparisons(self, three_series):
        result = compute_pairwise_ddtw(three_series, three_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_series):
        result = compute_pairwise_ddtw(three_series, three_series)
        assert result.height == 3

    def test_different_dataframes(self):
        """Cross-DataFrame comparison should work."""
        df1 = pl.DataFrame({
            "unique_id": ["X"] * 6,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        df2 = pl.DataFrame({
            "unique_id": ["Y"] * 6,
            "y": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result = compute_pairwise_ddtw(df1, df2)
        assert result.shape[0] == 1
        assert result["ddtw"][0] > 0

    def test_derivative_sensitivity(self):
        """DDTW should be sensitive to shape, not offset.
        A constant offset should not affect the derivative."""
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        shifted = [x + 100.0 for x in base]
        df = pl.DataFrame({
            "unique_id": ["A"] * 5 + ["B"] * 5,
            "y": base + shifted,
        })
        result = compute_pairwise_ddtw(df, df)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0  # same derivative

    def test_short_series_handled(self):
        """Series with < 3 points can't compute derivatives; DDTW returns inf."""
        df = pl.DataFrame({
            "unique_id": ["A"] * 2 + ["B"] * 2,
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        result = compute_pairwise_ddtw(df, df)
        d = _to_dict(result)
        # With only 2 points, derivative is empty → should return inf
        assert d[("A", "B")] == float("inf")
