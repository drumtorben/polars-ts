import polars as pl
import pytest

from polars_ts import compute_pairwise_sbd
from polars_ts.distance import compute_pairwise_distance

from .conftest import _to_dict


class TestSBDDirect:
    def test_identical_series(self, identical_series):
        result = compute_pairwise_sbd(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    def test_two_series(self, two_series):
        result = compute_pairwise_sbd(two_series, two_series)
        d = _to_dict(result)
        assert 0.0 < d[("A", "B")] < 2.0

    def test_three_series(self, three_series):
        result = compute_pairwise_sbd(three_series, three_series)
        d = _to_dict(result)
        assert len(d) == 3
        # A and B are similar, A and C are reversed
        assert d[("A", "B")] < d[("A", "C")]

    def test_negated_series_high_distance(self):
        df = pl.DataFrame({
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "y": [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        })
        result = compute_pairwise_sbd(df, df)
        d = _to_dict(result)
        # Negated series are anti-correlated, SBD should be > 1.0
        assert d[("A", "B")] > 1.0

    def test_sbd_range(self, two_series):
        result = compute_pairwise_sbd(two_series, two_series)
        d = _to_dict(result)
        for v in d.values():
            assert 0.0 <= v <= 2.0

    def test_int_ids(self, int_id_series):
        result = compute_pairwise_sbd(int_id_series, int_id_series)
        assert result.shape[0] == 1


class TestSBDUnifiedAPI:
    def test_via_dispatcher(self, two_series):
        result = compute_pairwise_distance(two_series, two_series, method="sbd")
        d = _to_dict(result)
        assert ("A", "B") in d

    def test_rejects_unknown_kwargs(self, two_series):
        with pytest.raises(ValueError, match="Unexpected"):
            compute_pairwise_distance(two_series, two_series, method="sbd", foo=1)
