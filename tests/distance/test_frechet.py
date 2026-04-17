import polars as pl
import pytest

from polars_ts import compute_pairwise_frechet
from polars_ts.distance import compute_pairwise_distance

from .conftest import _to_dict


class TestFrechetDirect:
    def test_identical_series(self, identical_series):
        result = compute_pairwise_frechet(identical_series, identical_series)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    def test_two_series(self, two_series):
        result = compute_pairwise_frechet(two_series, two_series)
        d = _to_dict(result)
        # Last points differ: 4 vs 5, so Frechet distance = 1.0
        assert d[("A", "B")] == pytest.approx(1.0, abs=1e-10)

    def test_three_series(self, three_series):
        result = compute_pairwise_frechet(three_series, three_series)
        d = _to_dict(result)
        assert len(d) == 3

    def test_known_value(self):
        """Frechet distance between [0, 1] and [0, 2] should be 1.0."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 2 + ["B"] * 2,
                "y": [0.0, 1.0, 0.0, 2.0],
            }
        )
        result = compute_pairwise_frechet(df, df)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(1.0, abs=1e-10)

    def test_constant_offset(self):
        """Series offset by constant c should have Frechet = c."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        result = compute_pairwise_frechet(df, df)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(5.0, abs=1e-10)

    def test_non_negative(self, two_series):
        result = compute_pairwise_frechet(two_series, two_series)
        d = _to_dict(result)
        for v in d.values():
            assert v >= 0.0

    def test_int_ids(self, int_id_series):
        result = compute_pairwise_frechet(int_id_series, int_id_series)
        assert result.shape[0] == 1


class TestFrechetUnifiedAPI:
    def test_via_dispatcher(self, two_series):
        result = compute_pairwise_distance(two_series, two_series, method="frechet")
        d = _to_dict(result)
        assert ("A", "B") in d

    def test_rejects_unknown_kwargs(self, two_series):
        with pytest.raises(ValueError, match="Unexpected"):
            compute_pairwise_distance(two_series, two_series, method="frechet", x=1)
