import polars as pl
from polars_ts_rs.polars_ts_rs import compute_pairwise_wdtw

from tests.distance.conftest import _to_dict


class TestWDTWBasic:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_wdtw(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance_positive(self, two_series):
        result = compute_pairwise_wdtw(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] > 0

    def test_output_columns(self, two_series):
        result = compute_pairwise_wdtw(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "wdtw"}

    def test_single_series_empty(self, single_series):
        result = compute_pairwise_wdtw(single_series, single_series)
        assert result.shape[0] == 0

    def test_three_series_pairs(self, three_series):
        result = compute_pairwise_wdtw(three_series, three_series)
        d = _to_dict(result)
        assert len(d) == 3

    def test_non_negativity(self, three_series):
        result = compute_pairwise_wdtw(three_series, three_series)
        assert (result["wdtw"] >= 0).all()

    def test_int_id_preserved(self, int_id_series):
        result = compute_pairwise_wdtw(int_id_series, int_id_series)
        assert result["id_1"].dtype == pl.Int64
        assert result["id_2"].dtype == pl.Int64


class TestWDTWProperties:
    def test_symmetry(self, three_series):
        df_a = three_series.filter(pl.col("unique_id") == "A")
        df_c = three_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_wdtw(df_a, df_c)
        ca = compute_pairwise_wdtw(df_c, df_a)
        assert abs(ac["wdtw"][0] - ca["wdtw"][0]) < 1e-10

    def test_no_self_comparisons(self, three_series):
        result = compute_pairwise_wdtw(three_series, three_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_series):
        result = compute_pairwise_wdtw(three_series, three_series)
        assert result.height == 3

    def test_g_parameter_affects_distance(self, two_series):
        """Different g values should produce different distances."""
        d_low = _to_dict(compute_pairwise_wdtw(two_series, two_series, g=0.01))
        d_high = _to_dict(compute_pairwise_wdtw(two_series, two_series, g=1.0))
        # Different penalty parameters should give different results
        assert d_low[("A", "B")] != d_high[("A", "B")]

    def test_default_g_works(self, two_series):
        """Calling without g should use default and not crash."""
        result = compute_pairwise_wdtw(two_series, two_series)
        assert result.shape[0] == 1

    def test_reversed_series_larger(self, three_series):
        """C (reversed A) should be further from A than B (differs by 1 point)."""
        result = compute_pairwise_wdtw(three_series, three_series)
        d = _to_dict(result)
        assert d[("A", "C")] > d[("A", "B")]

    def test_different_dataframes(self):
        df1 = pl.DataFrame(
            {
                "unique_id": ["X"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df2 = pl.DataFrame(
            {
                "unique_id": ["Y"] * 4,
                "y": [4.0, 3.0, 2.0, 1.0],
            }
        )
        result = compute_pairwise_wdtw(df1, df2)
        assert result.shape[0] == 1
        assert result["wdtw"][0] > 0
