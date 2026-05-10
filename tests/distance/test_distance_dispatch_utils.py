import polars as pl
import pytest

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict
from polars_ts_rs.polars_ts_rs import compute_pairwise_dtw, compute_pairwise_wdtw


class TestDistanceDispatchUtils:
    def test_compute_distances_dtw_matches_direct(self, two_series):
        dispatched = compute_distances(two_series, two_series, method="dtw")
        direct = compute_pairwise_dtw(two_series, two_series)
        assert dispatched["dtw"].to_list() == pytest.approx(direct["dtw"].to_list())

    def test_compute_distances_wdtw_kwargs_passthrough(self, shifted_series):
        dispatched = compute_distances(shifted_series, shifted_series, method="wdtw", g=0.01)
        direct = compute_pairwise_wdtw(shifted_series, shifted_series, g=0.01)
        assert dispatched["wdtw"].to_list() == pytest.approx(direct["wdtw"].to_list())

    def test_unknown_method_raises(self, two_series):
        with pytest.raises(ValueError, match=r"Unknown distance method"):
            compute_distances(two_series, two_series, method="nope")

    def test_unexpected_kwarg_raises(self, two_series):
        with pytest.raises(ValueError, match=r"Unexpected kwargs"):
            compute_distances(two_series, two_series, method="wdtw", epsilon=0.5)

    def test_pairwise_to_dict_is_symmetric(self, two_series):
        df = compute_distances(two_series, two_series, method="dtw")
        d = pairwise_to_dict(df)
        assert d[("A", "B")] == d[("B", "A")]

    def test_pairwise_to_dict_empty_df(self):
        df = pl.DataFrame({"id_1": [], "id_2": [], "dtw": []})
        assert pairwise_to_dict(df) == {}

    def test_pairwise_to_dict_coerces_int_ids_to_str(self):
        df = pl.DataFrame({"id_1": [1], "id_2": [2], "dtw": [0.5]})
        d = pairwise_to_dict(df)
        assert d[("1", "2")] == 0.5
        assert d[("2", "1")] == 0.5
