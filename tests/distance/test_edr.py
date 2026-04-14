import polars as pl
import pytest

from polars_ts import compute_pairwise_edr
from polars_ts.distance import compute_pairwise_distance

from .conftest import _to_dict


class TestEDRDirect:
    def test_identical_series(self, identical_series):
        result = compute_pairwise_edr(identical_series, identical_series, epsilon=0.5)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    def test_two_series_small_epsilon(self, two_series):
        result = compute_pairwise_edr(two_series, two_series, epsilon=0.5)
        d = _to_dict(result)
        # Only last point differs by 1.0, which is > epsilon=0.5
        assert d[("A", "B")] > 0.0

    def test_two_series_large_epsilon(self, two_series):
        result = compute_pairwise_edr(two_series, two_series, epsilon=1.5)
        d = _to_dict(result)
        # All points within epsilon=1.5, so distance should be 0
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    def test_three_series(self, three_series):
        result = compute_pairwise_edr(three_series, three_series, epsilon=0.5)
        d = _to_dict(result)
        assert len(d) == 3

    def test_normalized_range(self, two_series):
        result = compute_pairwise_edr(two_series, two_series, epsilon=0.1)
        d = _to_dict(result)
        for v in d.values():
            assert 0.0 <= v <= 1.0

    def test_completely_different(self):
        """Series with no matching points should have high EDR."""
        df = pl.DataFrame({
            "unique_id": ["A"] * 3 + ["B"] * 3,
            "y": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
        })
        result = compute_pairwise_edr(df, df, epsilon=0.01)
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(1.0, abs=1e-10)

    def test_default_epsilon(self, two_series):
        """Default epsilon should be 0.1."""
        result = compute_pairwise_edr(two_series, two_series)
        d = _to_dict(result)
        assert ("A", "B") in d

    def test_int_ids(self, int_id_series):
        result = compute_pairwise_edr(int_id_series, int_id_series, epsilon=0.5)
        assert result.shape[0] == 1


class TestEDRUnifiedAPI:
    def test_via_dispatcher(self, two_series):
        result = compute_pairwise_distance(two_series, two_series, method="edr", epsilon=0.5)
        d = _to_dict(result)
        assert ("A", "B") in d

    def test_epsilon_kwarg(self, identical_series):
        result = compute_pairwise_distance(
            identical_series, identical_series, method="edr", epsilon=0.5
        )
        d = _to_dict(result)
        assert d[("A", "B")] == pytest.approx(0.0, abs=1e-10)

    def test_rejects_unknown_kwargs(self, two_series):
        with pytest.raises(ValueError, match="Unexpected"):
            compute_pairwise_distance(two_series, two_series, method="edr", foo=1)
