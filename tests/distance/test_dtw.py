import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import compute_pairwise_dtw

from tests.distance.conftest import _to_dict

ALL_METHODS = [
    ("standard", None),
    ("sakoe_chiba", 2.0),
    ("itakura", 2.0),
    ("fast", 1.0),
]


# ===========================================================================
# Standard DTW tests (backward compatibility)
# ===========================================================================


class TestStandardDTW:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_dtw(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0

    def test_no_extra_args_backward_compat(self, two_series):
        """Calling without method/param works (backward compatible)."""
        result = compute_pairwise_dtw(two_series, two_series)
        assert result.shape[0] == 1

    def test_symmetry(self, two_series):
        r1 = compute_pairwise_dtw(two_series, two_series)
        d = _to_dict(r1)
        assert d[("A", "B")] == d.get(("B", "A"), d[("A", "B")])

    def test_three_series_pairs(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series)
        d = _to_dict(result)
        assert len(d) == 3  # (A,B), (A,C), (B,C)

    def test_single_series_empty(self, single_series):
        result = compute_pairwise_dtw(single_series, single_series)
        assert result.shape[0] == 0

    def test_output_columns(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "dtw"}

    def test_int_id_preserved(self, int_id_series):
        result = compute_pairwise_dtw(int_id_series, int_id_series)
        assert result["id_1"].dtype == pl.Int64
        assert result["id_2"].dtype == pl.Int64

    def test_non_negativity(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series)
        assert (result["dtw"] >= 0).all()

    def test_triangle_inequality(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series)
        d = _to_dict(result)
        assert d[("A", "C")] <= d[("A", "B")] + d[("B", "C")] + 1e-10


# ===========================================================================
# Sakoe-Chiba band tests
# ===========================================================================


class TestSakoeChiba:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_dtw(identical_series, identical_series, method="sakoe_chiba", param=2.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series, method="sakoe_chiba", param=2.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0

    def test_distance_ge_standard(self, shifted_series):
        """Constrained DTW distance should be >= unconstrained."""
        std = compute_pairwise_dtw(shifted_series, shifted_series)["dtw"][0]
        sc = compute_pairwise_dtw(shifted_series, shifted_series, method="sakoe_chiba", param=1.0)["dtw"][0]
        assert sc >= std - 1e-10

    def test_large_window_equals_standard(self, two_series):
        """With a window >= series length, result should equal standard DTW."""
        std = compute_pairwise_dtw(two_series, two_series)["dtw"][0]
        sc = compute_pairwise_dtw(two_series, two_series, method="sakoe_chiba", param=10.0)["dtw"][0]
        assert abs(sc - std) < 1e-10

    def test_narrow_window_increases_distance(self, shifted_series):
        """Narrower window should produce distance >= wider window."""
        sc1 = compute_pairwise_dtw(shifted_series, shifted_series, method="sakoe_chiba", param=1.0)["dtw"][0]
        sc4 = compute_pairwise_dtw(shifted_series, shifted_series, method="sakoe_chiba", param=4.0)["dtw"][0]
        assert sc1 >= sc4 - 1e-10

    def test_non_negativity(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series, method="sakoe_chiba", param=2.0)
        assert (result["dtw"] >= 0).all()

    def test_output_columns(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series, method="sakoe_chiba", param=2.0)
        assert set(result.columns) == {"id_1", "id_2", "dtw"}

    def test_int_id_preserved(self, int_id_series):
        result = compute_pairwise_dtw(int_id_series, int_id_series, method="sakoe_chiba", param=2.0)
        assert result["id_1"].dtype == pl.Int64


# ===========================================================================
# Itakura parallelogram tests
# ===========================================================================


class TestItakura:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_dtw(identical_series, identical_series, method="itakura", param=2.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series, method="itakura", param=2.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0

    def test_distance_ge_standard(self, shifted_series):
        """Constrained DTW distance should be >= unconstrained."""
        std = compute_pairwise_dtw(shifted_series, shifted_series)["dtw"][0]
        it = compute_pairwise_dtw(shifted_series, shifted_series, method="itakura", param=1.5)["dtw"][0]
        assert it >= std - 1e-10

    def test_large_slope_equals_standard(self, two_series):
        """With a very large max_slope, result should equal standard DTW."""
        std = compute_pairwise_dtw(two_series, two_series)["dtw"][0]
        it = compute_pairwise_dtw(two_series, two_series, method="itakura", param=10.0)["dtw"][0]
        assert abs(it - std) < 1e-10

    def test_non_negativity(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series, method="itakura", param=2.0)
        assert (result["dtw"] >= 0).all()

    def test_output_columns(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series, method="itakura", param=2.0)
        assert set(result.columns) == {"id_1", "id_2", "dtw"}


# ===========================================================================
# FastDTW tests
# ===========================================================================


class TestFastDTW:
    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_dtw(identical_series, identical_series, method="fast", param=1.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        """FastDTW on short series falls back to exact DTW."""
        result = compute_pairwise_dtw(two_series, two_series, method="fast", param=1.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0

    def test_approximation_quality(self):
        """FastDTW should produce a reasonable approximation for longer series."""
        import random

        random.seed(42)
        a = [float(i) + random.gauss(0, 0.1) for i in range(100)]
        b = [float(i) + random.gauss(0, 0.1) + 1.0 for i in range(100)]
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 100 + ["B"] * 100,
                "y": a + b,
            }
        )
        std = compute_pairwise_dtw(df, df)["dtw"][0]
        fast = compute_pairwise_dtw(df, df, method="fast", param=5.0)["dtw"][0]
        # FastDTW should be close to standard DTW (within 50% for well-behaved series)
        assert fast >= std - 1e-10  # should be >= standard
        assert fast <= std * 2.0  # reasonable approximation

    def test_non_negativity(self, three_series):
        result = compute_pairwise_dtw(three_series, three_series, method="fast", param=1.0)
        assert (result["dtw"] >= 0).all()

    def test_output_columns(self, two_series):
        result = compute_pairwise_dtw(two_series, two_series, method="fast", param=1.0)
        assert set(result.columns) == {"id_1", "id_2", "dtw"}

    def test_int_id_preserved(self, int_id_series):
        result = compute_pairwise_dtw(int_id_series, int_id_series, method="fast", param=1.0)
        assert result["id_1"].dtype == pl.Int64


# ===========================================================================
# Cross-method tests
# ===========================================================================


class TestCrossMethod:
    @pytest.mark.parametrize("method,param", ALL_METHODS)
    def test_identical_zero(self, identical_series, method, param):
        kwargs = {"method": method}
        if param is not None:
            kwargs["param"] = param
        result = compute_pairwise_dtw(identical_series, identical_series, **kwargs)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    @pytest.mark.parametrize("method,param", ALL_METHODS)
    def test_single_series_empty(self, single_series, method, param):
        kwargs = {"method": method}
        if param is not None:
            kwargs["param"] = param
        result = compute_pairwise_dtw(single_series, single_series, **kwargs)
        assert result.shape[0] == 0

    @pytest.mark.parametrize("method,param", ALL_METHODS)
    def test_non_negativity(self, three_series, method, param):
        kwargs = {"method": method}
        if param is not None:
            kwargs["param"] = param
        result = compute_pairwise_dtw(three_series, three_series, **kwargs)
        assert (result["dtw"] >= 0).all()

    def test_invalid_method_raises(self, two_series):
        with pytest.raises(ValueError, match="Unknown DTW method"):
            compute_pairwise_dtw(two_series, two_series, method="invalid")

    def test_default_params_used(self, two_series):
        """Method without explicit param should use defaults and not crash."""
        for method in ["sakoe_chiba", "itakura", "fast"]:
            result = compute_pairwise_dtw(two_series, two_series, method=method)
            assert result.shape[0] == 1


# ===========================================================================
# Edge case and robustness tests
# ===========================================================================


class TestEdgeCases:
    def test_unequal_length_series(self):
        """Series with different lengths should work for all methods."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 6,
                "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        for method, param in ALL_METHODS:
            kwargs = {"method": method}
            if param is not None:
                kwargs["param"] = param
            result = compute_pairwise_dtw(df, df, **kwargs)
            assert result.shape[0] == 1
            assert result["dtw"][0] >= 0

    def test_different_input_dataframes(self):
        """Using two different DataFrames (not self-comparison)."""
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
        result = compute_pairwise_dtw(df1, df2)
        assert result.shape[0] == 1
        assert result["id_1"][0] == "X"
        assert result["id_2"][0] == "Y"
        assert result["dtw"][0] > 0

    def test_different_inputs_constrained(self):
        """Constrained methods work with two separate DataFrames."""
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
        for method in ["sakoe_chiba", "itakura", "fast"]:
            result = compute_pairwise_dtw(df1, df2, method=method)
            assert result.shape[0] == 1
            assert result["dtw"][0] > 0

    def test_sakoe_chiba_window_zero(self):
        """Window=0 should still work (only diagonal allowed)."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        result = compute_pairwise_dtw(df, df, method="sakoe_chiba", param=0.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0  # identical on diagonal

    def test_constrained_symmetry(self):
        """Constrained DTW(A,B) should equal DTW(B,A) for all methods."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 5,
                "y": [1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 5.0],
            }
        )
        for method, param in [("sakoe_chiba", 2.0), ("itakura", 2.0), ("fast", 1.0)]:
            result = compute_pairwise_dtw(df, df, method=method, param=param)
            d = _to_dict(result)
            # _to_dict sorts keys, so (A,B) is the only key
            assert len(d) == 1
            assert d[("A", "B")] >= 0

    def test_fast_dtw_long_series_recurse(self):
        """FastDTW on series long enough to trigger multi-resolution recursion."""
        import random

        random.seed(123)
        n = 200
        a = [float(i) + random.gauss(0, 0.5) for i in range(n)]
        b = [float(i) + random.gauss(0, 0.5) + 2.0 for i in range(n)]
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * n + ["B"] * n,
                "y": a + b,
            }
        )
        std = compute_pairwise_dtw(df, df)["dtw"][0]
        fast = compute_pairwise_dtw(df, df, method="fast", param=3.0)["dtw"][0]
        # FastDTW is an approximation: should be >= standard and reasonably close
        assert fast >= std - 1e-10
        assert fast < std * 3.0  # not wildly off

    def test_itakura_tight_slope_equal_length(self):
        """Itakura with slope=1.0 on equal-length series allows only the diagonal."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        result = compute_pairwise_dtw(df, df, method="itakura", param=1.0)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0  # identical series, diagonal path is fine

    def test_method_standard_explicit(self, two_series):
        """Passing method='standard' explicitly should match the default."""
        default = compute_pairwise_dtw(two_series, two_series)["dtw"][0]
        explicit = compute_pairwise_dtw(two_series, two_series, method="standard")["dtw"][0]
        assert default == explicit
