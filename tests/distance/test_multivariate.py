import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_dtw_multi,
    compute_pairwise_msm_multi,
)

from tests.distance.conftest import _to_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_multi_series():
    """Two multivariate time series with 2 dimensions."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4,
        "y1": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0],
        "y2": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 0.0],
    })


@pytest.fixture
def three_multi_series():
    """Three multivariate time series with 2 dimensions."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "y1": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "y2": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def identical_multi_series():
    """Two identical multivariate time series."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4 + ["B"] * 4,
        "y1": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "y2": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
    })


@pytest.fixture
def single_multi_series():
    """A single multivariate time series."""
    return pl.DataFrame({
        "unique_id": ["A"] * 4,
        "y1": [1.0, 2.0, 3.0, 4.0],
        "y2": [4.0, 3.0, 2.0, 1.0],
    })


@pytest.fixture
def int_id_multi_series():
    """Multivariate time series with integer unique_id."""
    return pl.DataFrame({
        "unique_id": [1] * 4 + [2] * 4,
        "y1": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        "y2": [4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0],
    })


# ===========================================================================
# Multivariate DTW tests
# ===========================================================================

class TestMultivariateDTW:
    def test_identical_series_zero(self, identical_multi_series):
        result = compute_pairwise_dtw_multi(identical_multi_series, identical_multi_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance_positive(self, two_multi_series):
        result = compute_pairwise_dtw_multi(two_multi_series, two_multi_series)
        d = _to_dict(result)
        assert d[("A", "B")] > 0

    def test_output_columns(self, two_multi_series):
        result = compute_pairwise_dtw_multi(two_multi_series, two_multi_series)
        assert set(result.columns) == {"id_1", "id_2", "dtw_multi"}

    def test_single_series_empty(self, single_multi_series):
        result = compute_pairwise_dtw_multi(single_multi_series, single_multi_series)
        assert result.shape[0] == 0

    def test_three_series_pairs(self, three_multi_series):
        result = compute_pairwise_dtw_multi(three_multi_series, three_multi_series)
        d = _to_dict(result)
        assert len(d) == 3

    def test_non_negativity(self, three_multi_series):
        result = compute_pairwise_dtw_multi(three_multi_series, three_multi_series)
        assert (result["dtw_multi"] >= 0).all()

    def test_int_id_preserved(self, int_id_multi_series):
        result = compute_pairwise_dtw_multi(int_id_multi_series, int_id_multi_series)
        assert result["id_1"].dtype == pl.Int64
        assert result["id_2"].dtype == pl.Int64

    def test_no_self_comparisons(self, three_multi_series):
        result = compute_pairwise_dtw_multi(three_multi_series, three_multi_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_multi_series):
        result = compute_pairwise_dtw_multi(three_multi_series, three_multi_series)
        assert result.height == 3

    def test_manhattan_metric(self, two_multi_series):
        result = compute_pairwise_dtw_multi(two_multi_series, two_multi_series, metric="manhattan")
        assert result.shape[0] == 1
        assert result["dtw_multi"][0] > 0

    def test_euclidean_metric(self, two_multi_series):
        result = compute_pairwise_dtw_multi(two_multi_series, two_multi_series, metric="euclidean")
        assert result.shape[0] == 1
        assert result["dtw_multi"][0] > 0

    def test_metrics_differ(self, two_multi_series):
        """Manhattan and Euclidean should produce different distances."""
        d_man = _to_dict(compute_pairwise_dtw_multi(two_multi_series, two_multi_series, metric="manhattan"))
        d_euc = _to_dict(compute_pairwise_dtw_multi(two_multi_series, two_multi_series, metric="euclidean"))
        assert d_man[("A", "B")] != d_euc[("A", "B")]

    def test_default_metric_is_manhattan(self, two_multi_series):
        """Default (no metric arg) should equal explicit manhattan."""
        d_default = _to_dict(compute_pairwise_dtw_multi(two_multi_series, two_multi_series))
        d_manhattan = _to_dict(compute_pairwise_dtw_multi(two_multi_series, two_multi_series, metric="manhattan"))
        assert d_default[("A", "B")] == d_manhattan[("A", "B")]

    def test_symmetry(self, three_multi_series):
        df_a = three_multi_series.filter(pl.col("unique_id") == "A")
        df_c = three_multi_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_dtw_multi(df_a, df_c)
        ca = compute_pairwise_dtw_multi(df_c, df_a)
        assert abs(ac["dtw_multi"][0] - ca["dtw_multi"][0]) < 1e-10

    def test_three_dimensions(self):
        """Works with 3+ dimension columns."""
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 3,
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "y": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0],
            "z": [7.0, 8.0, 9.0, 9.0, 8.0, 7.0],
        })
        result = compute_pairwise_dtw_multi(df, df)
        assert result.shape[0] == 1
        assert result["dtw_multi"][0] > 0


# ===========================================================================
# Multivariate MSM tests
# ===========================================================================

class TestMultivariateMSM:
    def test_identical_series_zero(self, identical_multi_series):
        result = compute_pairwise_msm_multi(identical_multi_series, identical_multi_series)
        d = _to_dict(result)
        assert d[("A", "B")] == 0.0

    def test_basic_distance_positive(self, two_multi_series):
        result = compute_pairwise_msm_multi(two_multi_series, two_multi_series)
        d = _to_dict(result)
        assert d[("A", "B")] > 0

    def test_output_columns(self, two_multi_series):
        result = compute_pairwise_msm_multi(two_multi_series, two_multi_series)
        assert set(result.columns) == {"id_1", "id_2", "msm_multi"}

    def test_single_series_empty(self, single_multi_series):
        result = compute_pairwise_msm_multi(single_multi_series, single_multi_series)
        assert result.shape[0] == 0

    def test_three_series_pairs(self, three_multi_series):
        result = compute_pairwise_msm_multi(three_multi_series, three_multi_series)
        d = _to_dict(result)
        assert len(d) == 3

    def test_non_negativity(self, three_multi_series):
        result = compute_pairwise_msm_multi(three_multi_series, three_multi_series)
        assert (result["msm_multi"] >= 0).all()

    def test_int_id_preserved(self, int_id_multi_series):
        result = compute_pairwise_msm_multi(int_id_multi_series, int_id_multi_series)
        assert result["id_1"].dtype == pl.Int64
        assert result["id_2"].dtype == pl.Int64

    def test_no_self_comparisons(self, three_multi_series):
        result = compute_pairwise_msm_multi(three_multi_series, three_multi_series)
        for row in result.to_dicts():
            assert row["id_1"] != row["id_2"]

    def test_no_duplicate_pairs(self, three_multi_series):
        result = compute_pairwise_msm_multi(three_multi_series, three_multi_series)
        assert result.height == 3

    def test_c_parameter_affects_distance(self):
        """Different c values should produce different distances when inserts/deletes are needed."""
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 5,
            "y1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y2": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0, -1.0],
        })
        d_low = _to_dict(compute_pairwise_msm_multi(df, df, c=0.1))
        d_high = _to_dict(compute_pairwise_msm_multi(df, df, c=10.0))
        assert d_low[("A", "B")] != d_high[("A", "B")]

    def test_default_c_works(self, two_multi_series):
        result = compute_pairwise_msm_multi(two_multi_series, two_multi_series)
        assert result.shape[0] == 1

    def test_symmetry(self, three_multi_series):
        df_a = three_multi_series.filter(pl.col("unique_id") == "A")
        df_c = three_multi_series.filter(pl.col("unique_id") == "C")
        ac = compute_pairwise_msm_multi(df_a, df_c)
        ca = compute_pairwise_msm_multi(df_c, df_a)
        assert abs(ac["msm_multi"][0] - ca["msm_multi"][0]) < 1e-10

    def test_three_dimensions(self):
        """Works with 3+ dimension columns."""
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 3,
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "y": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0],
            "z": [7.0, 8.0, 9.0, 9.0, 8.0, 7.0],
        })
        result = compute_pairwise_msm_multi(df, df)
        assert result.shape[0] == 1
        assert result["msm_multi"][0] > 0
