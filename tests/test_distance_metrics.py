import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_dtw,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_twe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_series():
    """Two simple time series A and B that differ by one point."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4,
        "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0],
    })


@pytest.fixture
def three_series():
    """Three time series: A ascending, B similar to A, C reversed."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })


@pytest.fixture
def identical_series():
    """Two identical time series."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4,
        "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def single_series():
    """A single time series — no pairs to compare."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4,
        "y": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def int_id_series():
    """Time series with integer unique_id."""
    return pl.DataFrame({
        "unique_id": [1] * 4 + [2] * 4,
        "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(df: pl.DataFrame) -> dict:
    """Convert pairwise result to {(id1, id2): distance} dict, sorted keys."""
    rows = df.to_dicts()
    dist_col = [c for c in df.columns if c not in ("id_1", "id_2")][0]
    result = {}
    for r in rows:
        key = tuple(sorted([str(r["id_1"]), str(r["id_2"])]))
        result[key] = r[dist_col]
    return result


# ===========================================================================
# ERP tests
# ===========================================================================

class TestERP:
    def test_identical_series_zero_distance(self, identical_series):
        result = compute_pairwise_erp(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        result = compute_pairwise_erp(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0  # only last point differs by 1

    def test_symmetric(self, three_series):
        result = compute_pairwise_erp(three_series, three_series)
        d = _to_dict(result)
        assert d[("A", "C")] == d[("A", "C")]  # trivially true
        # Also check A-C computed from both directions gives same result
        df_a = three_series.filter(pl.col("unique_id") == "A")
        df_c = three_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_erp(df_a, df_c)
        ca = compute_pairwise_erp(df_c, df_a)
        assert ac["erp"][0] == ca["erp"][0]

    def test_gap_value_affects_distance(self):
        # Use series of different lengths to force insertions/deletions
        # where the gap value matters
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 5,
            "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        })
        d0 = _to_dict(compute_pairwise_erp(df, df, g=0.0))
        d5 = _to_dict(compute_pairwise_erp(df, df, g=5.0))
        # Different gap values should produce different distances
        # when insertions/deletions are needed
        assert d0[("A", "B")] != d5[("A", "B")]

    def test_triangle_inequality(self, three_series):
        result = compute_pairwise_erp(three_series, three_series)
        d = _to_dict(result)
        # ERP is a metric: d(A,C) <= d(A,B) + d(B,C)
        assert d[("A", "C")] <= d[("A", "B")] + d[("B", "C")] + 1e-10

    def test_output_columns(self, two_series):
        result = compute_pairwise_erp(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "erp"}

    def test_no_self_comparisons(self, three_series):
        result = compute_pairwise_erp(three_series, three_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_series):
        result = compute_pairwise_erp(three_series, three_series)
        # 3 series → 3 unique pairs
        assert result.height == 3

    def test_single_series_empty_result(self, single_series):
        result = compute_pairwise_erp(single_series, single_series)
        assert result.height == 0

    def test_preserves_int_dtype(self, int_id_series):
        result = compute_pairwise_erp(int_id_series, int_id_series)
        assert result["id_1"].dtype == result["id_2"].dtype
        assert result["id_1"].dtype != pl.String

    def test_non_negative(self, three_series):
        result = compute_pairwise_erp(three_series, three_series)
        assert (result["erp"] >= 0).all()


# ===========================================================================
# LCSS tests
# ===========================================================================

class TestLCSS:
    def test_identical_series_zero_distance(self, identical_series):
        result = compute_pairwise_lcss(identical_series, identical_series, epsilon=0.01)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance(self, two_series):
        result = compute_pairwise_lcss(two_series, two_series, epsilon=0.5)
        d = _to_dict(result)
        # A=[1,2,3,4] B=[1,2,3,5]: 3 of 4 match within 0.5 → distance = 1-3/4 = 0.25
        assert d[("A", "B")] == 0.25

    def test_large_epsilon_all_match(self, three_series):
        result = compute_pairwise_lcss(three_series, three_series, epsilon=100.0)
        d = _to_dict(result)
        # With huge epsilon everything matches → distance = 0
        for v in d.values():
            assert v == 0.0

    def test_tiny_epsilon_no_match(self):
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 3,
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        result = compute_pairwise_lcss(df, df, epsilon=0.001)
        d = _to_dict(result)
        assert d[("A", "B")] == 1.0  # no matches → max distance

    def test_range_zero_to_one(self, three_series):
        result = compute_pairwise_lcss(three_series, three_series)
        for v in result["lcss"].to_list():
            assert 0.0 <= v <= 1.0

    def test_default_epsilon(self, two_series):
        # Default epsilon=1.0: A=[1,2,3,4] B=[1,2,3,5], all within 1.0
        result = compute_pairwise_lcss(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_output_columns(self, two_series):
        result = compute_pairwise_lcss(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "lcss"}

    def test_no_self_comparisons(self, three_series):
        result = compute_pairwise_lcss(three_series, three_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_series):
        result = compute_pairwise_lcss(three_series, three_series)
        assert result.height == 3

    def test_single_series_empty_result(self, single_series):
        result = compute_pairwise_lcss(single_series, single_series)
        assert result.height == 0

    def test_preserves_int_dtype(self, int_id_series):
        result = compute_pairwise_lcss(int_id_series, int_id_series)
        assert result["id_1"].dtype == result["id_2"].dtype
        assert result["id_1"].dtype != pl.String

    def test_symmetric(self, three_series):
        df_a = three_series.filter(pl.col("unique_id") == "A")
        df_c = three_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_lcss(df_a, df_c, epsilon=1.0)
        ca = compute_pairwise_lcss(df_c, df_a, epsilon=1.0)
        assert ac["lcss"][0] == ca["lcss"][0]


# ===========================================================================
# TWE tests
# ===========================================================================

class TestTWE:
    def test_identical_series_zero_distance(self, identical_series):
        result = compute_pairwise_twe(identical_series, identical_series)
        d = _to_dict(result)
        assert abs(d[("A", "B")]) < 1e-10

    def test_basic_distance(self, two_series):
        result = compute_pairwise_twe(two_series, two_series)
        d = _to_dict(result)
        assert d[("A", "B")] > 0  # not identical → positive distance

    def test_reversed_series_larger(self, three_series):
        result = compute_pairwise_twe(three_series, three_series)
        d = _to_dict(result)
        # C is A reversed — should be further than B which differs by 1 point
        assert d[("A", "C")] > d[("A", "B")]

    def test_stiffness_increases_distance(self, two_series):
        d_low = _to_dict(compute_pairwise_twe(two_series, two_series, 0.001, 1.0))
        d_high = _to_dict(compute_pairwise_twe(two_series, two_series, 10.0, 1.0))
        # Higher stiffness penalizes warping more
        assert d_high[("A", "B")] >= d_low[("A", "B")]

    def test_lambda_increases_distance(self, two_series):
        d_low = _to_dict(compute_pairwise_twe(two_series, two_series, 0.001, 0.1))
        d_high = _to_dict(compute_pairwise_twe(two_series, two_series, 0.001, 10.0))
        # Higher lambda penalizes insertions/deletions more
        assert d_high[("A", "B")] >= d_low[("A", "B")]

    def test_output_columns(self, two_series):
        result = compute_pairwise_twe(two_series, two_series)
        assert set(result.columns) == {"id_1", "id_2", "twe"}

    def test_no_self_comparisons(self, three_series):
        result = compute_pairwise_twe(three_series, three_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_series):
        result = compute_pairwise_twe(three_series, three_series)
        assert result.height == 3

    def test_single_series_empty_result(self, single_series):
        result = compute_pairwise_twe(single_series, single_series)
        assert result.height == 0

    def test_preserves_int_dtype(self, int_id_series):
        result = compute_pairwise_twe(int_id_series, int_id_series)
        assert result["id_1"].dtype == result["id_2"].dtype
        assert result["id_1"].dtype != pl.String

    def test_non_negative(self, three_series):
        result = compute_pairwise_twe(three_series, three_series)
        assert (result["twe"] >= 0).all()

    def test_symmetric(self, three_series):
        df_a = three_series.filter(pl.col("unique_id") == "A")
        df_c = three_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_twe(df_a, df_c)
        ca = compute_pairwise_twe(df_c, df_a)
        assert abs(ac["twe"][0] - ca["twe"][0]) < 1e-10


# ===========================================================================
# Cross-metric consistency tests
# ===========================================================================

class TestCrossMetric:
    def test_identical_all_zero(self, identical_series):
        """All metrics should return 0 for identical series."""
        dtw = _to_dict(compute_pairwise_dtw(identical_series, identical_series))
        erp = _to_dict(compute_pairwise_erp(identical_series, identical_series))
        lcss = _to_dict(compute_pairwise_lcss(identical_series, identical_series, epsilon=0.01))
        twe = _to_dict(compute_pairwise_twe(identical_series, identical_series))

        assert dtw[("A", "B")] == 0.0
        assert erp[("A", "B")] == 0.0
        assert lcss[("A", "B")] == 0.0
        assert abs(twe[("A", "B")]) < 1e-10

    def test_ordering_consistent(self, three_series):
        """All metrics should agree that A-B is closer than A-C."""
        dtw = _to_dict(compute_pairwise_dtw(three_series, three_series))
        erp = _to_dict(compute_pairwise_erp(three_series, three_series))
        lcss = _to_dict(compute_pairwise_lcss(three_series, three_series, epsilon=0.5))
        twe = _to_dict(compute_pairwise_twe(three_series, three_series))

        assert dtw[("A", "B")] < dtw[("A", "C")]
        assert erp[("A", "B")] < erp[("A", "C")]
        assert lcss[("A", "B")] < lcss[("A", "C")]
        assert twe[("A", "B")] < twe[("A", "C")]

    def test_all_same_pair_count(self, three_series):
        """All metrics should produce the same number of pairs."""
        n_dtw = compute_pairwise_dtw(three_series, three_series).height
        n_erp = compute_pairwise_erp(three_series, three_series).height
        n_lcss = compute_pairwise_lcss(three_series, three_series).height
        n_twe = compute_pairwise_twe(three_series, three_series).height
        assert n_dtw == n_erp == n_lcss == n_twe == 3

    def test_cross_dataframe(self):
        """Test comparing series across two different DataFrames."""
        df1 = pl.DataFrame({
            "unique_id": ["X"] * 3,
            "y": [1.0, 2.0, 3.0],
        })
        df2 = pl.DataFrame({
            "unique_id": ["Y"] * 3,
            "y": [3.0, 2.0, 1.0],
        })
        erp = compute_pairwise_erp(df1, df2)
        lcss = compute_pairwise_lcss(df1, df2)
        twe = compute_pairwise_twe(df1, df2)

        assert erp.height == 1
        assert lcss.height == 1
        assert twe.height == 1
        assert erp["erp"][0] > 0
        assert twe["twe"][0] > 0
