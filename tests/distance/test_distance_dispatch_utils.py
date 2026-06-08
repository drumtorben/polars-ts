import polars as pl

from polars_ts._distance_dispatch import pairwise_to_dict


class TestDistanceDispatchUtils:
    def test_pairwise_to_dict_is_symmetric(self):
        df = pl.DataFrame({"id_1": ["A", "B"], "id_2": ["B", "A"], "dtw": [1.0, 1.0]})
        d = pairwise_to_dict(df)
        assert d[("A", "B")] == d[("B", "A")] == 1.0

    def test_pairwise_to_dict_empty_df(self):
        df = pl.DataFrame({"id_1": [], "id_2": [], "dtw": []})
        assert pairwise_to_dict(df) == {}

    def test_pairwise_to_dict_coerces_int_ids_to_str(self):
        df = pl.DataFrame({"id_1": [1], "id_2": [2], "dtw": [0.5]})
        d = pairwise_to_dict(df)
        assert d[("1", "2")] == 0.5
        assert d[("2", "1")] == 0.5
