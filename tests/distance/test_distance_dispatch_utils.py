import polars as pl
import pytest

import polars_ts._distance_dispatch as dispatch
from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


class TestDistanceDispatchUtils:
    def test_compute_distances_dtw_dispatches(self, monkeypatch, two_series):
        calls = []

        def fake(df1, df2, **kwargs):
            calls.append((df1, df2, kwargs))
            return pl.DataFrame({"id_1": ["A"], "id_2": ["B"], "dtw": [1.25]})

        monkeypatch.setitem(dispatch._DISTANCE_FUNCS, "dtw", fake)

        result = compute_distances(two_series, two_series, method="dtw")

        assert calls == [(two_series, two_series, {})]
        assert result["dtw"].to_list() == [1.25]

    def test_compute_distances_wdtw_kwargs_passthrough(self, monkeypatch, shifted_series):
        calls = []

        def fake(df1, df2, **kwargs):
            calls.append((df1, df2, kwargs))
            return pl.DataFrame({"id_1": ["A"], "id_2": ["B"], "wdtw": [2.5]})

        monkeypatch.setitem(dispatch._DISTANCE_FUNCS, "wdtw", fake)

        result = compute_distances(shifted_series, shifted_series, method="wdtw", g=0.01)

        assert calls == [(shifted_series, shifted_series, {"g": 0.01})]
        assert result["wdtw"].to_list() == [2.5]

    def test_unknown_method_raises(self, two_series):
        with pytest.raises(ValueError, match=r"Unknown distance method"):
            compute_distances(two_series, two_series, method="nope")

    def test_unexpected_kwarg_raises(self, two_series):
        with pytest.raises(ValueError, match=r"Unexpected kwargs"):
            compute_distances(two_series, two_series, method="wdtw", epsilon=0.5)

    def test_pairwise_to_dict_is_symmetric(self, two_series):
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
