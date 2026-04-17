"""Tests for the unified compute_pairwise_distance API."""

import polars as pl
import pytest

from polars_ts import (
    compute_pairwise_ddtw,
    compute_pairwise_distance,
    compute_pairwise_dtw,
    compute_pairwise_edr,
    compute_pairwise_erp,
    compute_pairwise_frechet,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_sbd,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)


class TestUnifiedDispatch:
    """Verify that the unified API dispatches correctly to each metric."""

    def test_dtw_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="dtw")
        direct = compute_pairwise_dtw(two_series, two_series)
        assert unified["dtw"].to_list() == pytest.approx(direct["dtw"].to_list())

    def test_ddtw_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="ddtw")
        direct = compute_pairwise_ddtw(two_series, two_series)
        assert unified["ddtw"].to_list() == pytest.approx(direct["ddtw"].to_list())

    def test_wdtw_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="wdtw")
        direct = compute_pairwise_wdtw(two_series, two_series)
        assert unified["wdtw"].to_list() == pytest.approx(direct["wdtw"].to_list())

    def test_msm_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="msm")
        direct = compute_pairwise_msm(two_series, two_series)
        assert unified["msm"].to_list() == pytest.approx(direct["msm"].to_list())

    def test_erp_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="erp")
        direct = compute_pairwise_erp(two_series, two_series)
        assert unified["erp"].to_list() == pytest.approx(direct["erp"].to_list())

    def test_lcss_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="lcss")
        direct = compute_pairwise_lcss(two_series, two_series)
        assert unified["lcss"].to_list() == pytest.approx(direct["lcss"].to_list())

    def test_twe_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="twe")
        direct = compute_pairwise_twe(two_series, two_series)
        assert unified["twe"].to_list() == pytest.approx(direct["twe"].to_list())

    def test_sbd_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="sbd")
        direct = compute_pairwise_sbd(two_series, two_series)
        assert unified["sbd"].to_list() == pytest.approx(direct["sbd"].to_list())

    def test_frechet_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="frechet")
        direct = compute_pairwise_frechet(two_series, two_series)
        assert unified["frechet"].to_list() == pytest.approx(direct["frechet"].to_list())

    def test_edr_matches_direct(self, two_series):
        unified = compute_pairwise_distance(two_series, two_series, method="edr", epsilon=0.5)
        direct = compute_pairwise_edr(two_series, two_series, epsilon=0.5)
        assert unified["edr"].to_list() == pytest.approx(direct["edr"].to_list())


class TestUnifiedKwargs:
    """Verify that keyword arguments are passed through and affect the result."""

    @pytest.fixture
    def divergent_series(self):
        """Series designed to show parameter sensitivity in distance metrics."""
        return pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8,
                "y": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_wdtw_g_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="wdtw", g=0.01)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="wdtw", g=10.0)
        assert r1["wdtw"].to_list() != pytest.approx(r2["wdtw"].to_list(), abs=1e-6)

    def test_msm_c_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="msm", c=0.1)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="msm", c=100.0)
        assert r1["msm"].to_list() != pytest.approx(r2["msm"].to_list(), abs=1e-6)

    def test_erp_g_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="erp", g=0.0)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="erp", g=100.0)
        assert r1["erp"].to_list() != pytest.approx(r2["erp"].to_list(), abs=1e-6)

    def test_lcss_epsilon_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="lcss", epsilon=0.01)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="lcss", epsilon=100.0)
        assert r1["lcss"].to_list() != pytest.approx(r2["lcss"].to_list(), abs=1e-6)

    def test_twe_nu_changes_result(self, divergent_series):
        """Cover the nu parameter passthrough (distance.py line 144)."""
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="twe", nu=0.001)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="twe", nu=10.0)
        assert r1["twe"].to_list() != pytest.approx(r2["twe"].to_list(), abs=1e-6)

    def test_edr_epsilon_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="edr", epsilon=0.01)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="edr", epsilon=100.0)
        assert r1["edr"].to_list() != pytest.approx(r2["edr"].to_list(), abs=1e-6)

    def test_twe_lambda_changes_result(self, divergent_series):
        r1 = compute_pairwise_distance(divergent_series, divergent_series, method="twe", lambda_=0.001)
        r2 = compute_pairwise_distance(divergent_series, divergent_series, method="twe", lambda_=100.0)
        assert r1["twe"].to_list() != pytest.approx(r2["twe"].to_list(), abs=1e-6)

    def test_dtw_with_sakoe_chiba(self, two_series):
        r_std = compute_pairwise_distance(two_series, two_series, method="dtw")
        r_sc = compute_pairwise_distance(two_series, two_series, method="dtw", dtw_method="sakoe_chiba", param=1.0)
        # Constrained DTW distance >= unconstrained
        assert r_sc["dtw"].to_list()[0] >= r_std["dtw"].to_list()[0] - 1e-10

    def test_dtw_with_itakura(self, two_series):
        result = compute_pairwise_distance(two_series, two_series, method="dtw", dtw_method="itakura", param=2.0)
        assert len(result) > 0
        assert result["dtw"][0] >= 0

    def test_dtw_with_fast(self, two_series):
        result = compute_pairwise_distance(two_series, two_series, method="dtw", dtw_method="fast", param=1.0)
        assert len(result) > 0
        assert result["dtw"][0] >= 0

    def test_no_kwargs_uses_defaults(self, two_series):
        """Calling without kwargs should produce same result as direct call with defaults."""
        unified = compute_pairwise_distance(two_series, two_series, method="wdtw")
        direct = compute_pairwise_wdtw(two_series, two_series)
        assert unified["wdtw"].to_list() == pytest.approx(direct["wdtw"].to_list())


class TestUnifiedDefaults:
    """Verify default behavior."""

    def test_default_method_is_dtw(self, two_series):
        result = compute_pairwise_distance(two_series, two_series)
        assert "dtw" in result.columns

    def test_identical_series_zero(self, identical_series):
        result = compute_pairwise_distance(identical_series, identical_series)
        assert result["dtw"].to_list() == pytest.approx([0.0])

    def test_single_series_empty(self, single_series):
        result = compute_pairwise_distance(single_series, single_series)
        assert len(result) == 0


class TestUnifiedErrors:
    """Verify error handling."""

    def test_unknown_method(self, two_series):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_pairwise_distance(two_series, two_series, method="invalid")

    def test_unexpected_kwarg(self, two_series):
        with pytest.raises(ValueError, match="Unexpected keyword"):
            compute_pairwise_distance(two_series, two_series, method="ddtw", g=0.5)

    def test_unexpected_kwarg_dtw(self, two_series):
        with pytest.raises(ValueError, match="Unexpected keyword"):
            compute_pairwise_distance(two_series, two_series, method="dtw", epsilon=0.5)


class TestUnifiedMultivariate:
    """Verify multivariate dispatch."""

    @pytest.fixture
    def multi_series(self):
        return pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
                "z": [2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0],
            }
        )

    def test_dtw_multi(self, multi_series):
        result = compute_pairwise_distance(multi_series, multi_series, method="dtw_multi")
        assert len(result) > 0

    def test_msm_multi(self, multi_series):
        result = compute_pairwise_distance(multi_series, multi_series, method="msm_multi")
        assert len(result) > 0

    def test_dtw_multi_with_metric(self, multi_series):
        result = compute_pairwise_distance(multi_series, multi_series, method="dtw_multi", metric="euclidean")
        assert len(result) > 0
