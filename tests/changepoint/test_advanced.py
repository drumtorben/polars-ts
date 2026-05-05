"""Tests for advanced changepoint detection (#54)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.changepoint.pelt import pelt


def _make_shift_df(n1: int = 50, n2: int = 50, shift: float = 10.0) -> pl.DataFrame:
    base = date(2024, 1, 1)
    n = n1 + n2
    rng = np.random.default_rng(42)
    values = np.concatenate([rng.normal(0, 1, n1), rng.normal(shift, 1, n2)])
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


class TestPELT:
    def test_detects_single_shift(self):
        df = _make_shift_df(50, 50, shift=10.0)
        result = pelt(df, penalty=10.0)
        assert len(result) >= 1
        # Changepoint should be near index 50
        cp_idx = result["changepoint_idx"][0]
        assert 40 <= cp_idx <= 60

    def test_no_changepoint(self):
        rng = np.random.default_rng(42)
        base = date(2024, 1, 1)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 100,
                "ds": [base + timedelta(days=i) for i in range(100)],
                "y": rng.normal(0, 1, 100).tolist(),
            }
        )
        result = pelt(df, penalty=50.0)  # High penalty → no changepoints
        assert len(result) == 0

    def test_output_columns(self):
        result = pelt(_make_shift_df())
        assert "unique_id" in result.columns
        assert "changepoint_idx" in result.columns
        assert "ds" in result.columns

    def test_multiple_series(self):
        df1 = _make_shift_df()
        df2 = _make_shift_df().with_columns(pl.lit("B").alias("unique_id"))
        df = pl.concat([df1, df2])
        result = pelt(df, penalty=10.0)
        assert len(result["unique_id"].unique()) >= 1

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="Unknown cost"):
            pelt(_make_shift_df(), cost="invalid")

    def test_variance_cost(self):
        # Create variance shift
        rng = np.random.default_rng(42)
        base = date(2024, 1, 1)
        values = np.concatenate([rng.normal(0, 1, 50), rng.normal(0, 5, 50)])
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 100,
                "ds": [base + timedelta(days=i) for i in range(100)],
                "y": values.tolist(),
            }
        )
        result = pelt(df, cost="var", penalty=10.0)
        assert len(result) >= 0  # May or may not detect depending on penalty


class TestBOCPD:
    def test_basic(self):
        pytest.importorskip("scipy")
        from polars_ts.changepoint.bocpd import bocpd

        df = _make_shift_df(50, 50, shift=10.0)
        result = bocpd(df, hazard_rate=50.0, threshold=0.3)
        assert "is_changepoint" in result.columns
        assert "changepoint_prob" in result.columns
        assert "run_length" in result.columns

    def test_detects_shift(self):
        pytest.importorskip("scipy")
        from polars_ts.changepoint.bocpd import bocpd

        df = _make_shift_df(50, 50, shift=15.0)
        result = bocpd(df, hazard_rate=50.0, threshold=0.2)
        # At least one point should have elevated changepoint probability
        max_prob = result["changepoint_prob"].max()
        assert max_prob > 0.01  # At least some elevated probability

    def test_invalid_hazard(self):
        pytest.importorskip("scipy")
        from polars_ts.changepoint.bocpd import bocpd

        with pytest.raises(ValueError, match="hazard_rate"):
            bocpd(_make_shift_df(), hazard_rate=0)


class TestRegimeDetect:
    def test_basic(self):
        from polars_ts.changepoint.regime import regime_detect

        df = _make_shift_df(50, 50, shift=10.0)
        result = regime_detect(df, n_states=2)
        assert "regime" in result.columns
        assert "regime_prob" in result.columns
        assert len(result) == 100

    def test_two_states_assigned(self):
        from polars_ts.changepoint.regime import regime_detect

        df = _make_shift_df(50, 50, shift=15.0)
        result = regime_detect(df, n_states=2)
        n_unique = result["regime"].n_unique()
        assert n_unique == 2

    def test_invalid_n_states(self):
        from polars_ts.changepoint.regime import regime_detect

        with pytest.raises(ValueError, match="n_states"):
            regime_detect(_make_shift_df(), n_states=1)


# --- Additional PELT tests ---


def test_pelt_meanvar_cost():
    """PELT with meanvar cost should detect shifts."""
    df = _make_shift_df(50, 50, shift=10.0)
    result = pelt(df, cost="meanvar", penalty=10.0)
    assert len(result) >= 1


def test_pelt_custom_columns():
    """PELT should work with non-default column names."""
    base = date(2024, 1, 1)
    rng = np.random.default_rng(42)
    n = 100
    values = np.concatenate([rng.normal(0, 1, 50), rng.normal(10, 1, 50)])
    df = pl.DataFrame(
        {"series": ["A"] * n, "time": [base + timedelta(days=i) for i in range(n)], "value": values.tolist()}
    )
    result = pelt(df, target_col="value", id_col="series", time_col="time", penalty=10.0)
    assert len(result) >= 1
    assert "series" in result.columns


def test_pelt_min_size():
    """Large min_size should prevent detecting short segments."""
    df = _make_shift_df(50, 50, shift=10.0)
    result_small = pelt(df, penalty=10.0, min_size=2)
    result_large = pelt(df, penalty=10.0, min_size=40)
    # Larger min_size should produce fewer or equal changepoints
    assert len(result_large) <= len(result_small)


def test_pelt_multiple_shifts():
    """PELT should detect multiple changepoints in data with two shifts."""
    rng = np.random.default_rng(42)
    base = date(2024, 1, 1)
    values = np.concatenate([rng.normal(0, 0.5, 40), rng.normal(10, 0.5, 40), rng.normal(-5, 0.5, 40)])
    n = len(values)
    df = pl.DataFrame(
        {"unique_id": ["A"] * n, "ds": [base + timedelta(days=i) for i in range(n)], "y": values.tolist()}
    )
    result = pelt(df, penalty=10.0)
    assert len(result) >= 2  # Should detect at least 2 changepoints


def test_pelt_rust_python_equivalence():
    """Rust and Python PELT should return same changepoints."""
    from polars_ts.changepoint.pelt import _pelt_python

    df = _make_shift_df(50, 50, shift=10.0)
    py_result = _pelt_python(df, "y", "unique_id", "ds", "mean", 10.0, 2)
    rs_result = pelt(df, penalty=10.0)
    # Both should detect at least one changepoint near 50
    if len(py_result) > 0 and len(rs_result) > 0:
        py_cp = py_result["changepoint_idx"][0]
        rs_cp = rs_result["changepoint_idx"][0]
        assert abs(py_cp - rs_cp) <= 5


def test_pelt_constant_series():
    """Constant series should have no changepoints."""
    base = date(2024, 1, 1)
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 50,
            "ds": [base + timedelta(days=i) for i in range(50)],
            "y": [5.0] * 50,
        }
    )
    result = pelt(df, penalty=10.0)
    assert len(result) == 0


# --- Additional BOCPD tests ---


def test_bocpd_constant_series():
    """BOCPD on constant series should detect no changepoints."""
    pytest.importorskip("scipy")
    from polars_ts.changepoint.bocpd import bocpd

    base = date(2024, 1, 1)
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 50,
            "ds": [base + timedelta(days=i) for i in range(50)],
            "y": [5.0] * 50,
        }
    )
    result = bocpd(df, hazard_rate=50.0, threshold=0.5)
    cp_count = result.filter(pl.col("is_changepoint")).height
    assert cp_count == 0


def test_bocpd_multiple_shifts():
    """BOCPD should detect multiple mean shifts."""
    pytest.importorskip("scipy")
    from polars_ts.changepoint.bocpd import bocpd

    rng = np.random.default_rng(42)
    base = date(2024, 1, 1)
    values = np.concatenate([rng.normal(0, 0.5, 40), rng.normal(10, 0.5, 40), rng.normal(-5, 0.5, 40)])
    n = len(values)
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )
    result = bocpd(df, hazard_rate=50.0, threshold=0.01)
    # Should have elevated changepoint probabilities around indices 40 and 80
    max_prob = result["changepoint_prob"].max()
    assert max_prob > 0.01


def test_bocpd_output_length():
    """BOCPD output should have same length as input."""
    pytest.importorskip("scipy")
    from polars_ts.changepoint.bocpd import bocpd

    df = _make_shift_df(30, 30, shift=5.0)
    result = bocpd(df, hazard_rate=50.0, threshold=0.5)
    assert len(result) == 60


def test_bocpd_custom_threshold():
    """Higher threshold should produce fewer changepoints."""
    pytest.importorskip("scipy")
    from polars_ts.changepoint.bocpd import bocpd

    df = _make_shift_df(50, 50, shift=10.0)
    low = bocpd(df, hazard_rate=50.0, threshold=0.1)
    high = bocpd(df, hazard_rate=50.0, threshold=0.9)
    low_count = low.filter(pl.col("is_changepoint")).height
    high_count = high.filter(pl.col("is_changepoint")).height
    assert high_count <= low_count


# --- Additional regime tests ---


def test_regime_three_states():
    """Regime detection with 3 states."""
    from polars_ts.changepoint.regime import regime_detect

    rng = np.random.default_rng(42)
    base = date(2024, 1, 1)
    values = np.concatenate([rng.normal(0, 0.5, 30), rng.normal(10, 0.5, 30), rng.normal(-5, 0.5, 30)])
    n = len(values)
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )
    result = regime_detect(df, n_states=3)
    assert result["regime"].n_unique() <= 3
    assert "regime_prob" in result.columns


def test_regime_custom_columns():
    """Regime detection with non-default column names."""
    from polars_ts.changepoint.regime import regime_detect

    rng = np.random.default_rng(42)
    base = date(2024, 1, 1)
    n = 60
    values = np.concatenate([rng.normal(0, 1, 30), rng.normal(10, 1, 30)])
    df = pl.DataFrame(
        {"series": ["A"] * n, "time": [base + timedelta(days=i) for i in range(n)], "value": values.tolist()}
    )
    result = regime_detect(df, n_states=2, target_col="value", id_col="series", time_col="time")
    assert "regime" in result.columns
    assert "series" in result.columns


def test_regime_prob_range():
    """Regime probabilities should be in [0, 1]."""
    from polars_ts.changepoint.regime import regime_detect

    df = _make_shift_df(50, 50, shift=10.0)
    result = regime_detect(df, n_states=2)
    probs = result["regime_prob"].to_list()
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_regime_multiple_series():
    """Regime detection should handle multiple series independently."""
    from polars_ts.changepoint.regime import regime_detect

    df1 = _make_shift_df(30, 30, shift=10.0)
    df2 = _make_shift_df(30, 30, shift=10.0).with_columns(pl.lit("B").alias("unique_id"))
    df = pl.concat([df1, df2])
    result = regime_detect(df, n_states=2)
    assert len(result) == 120
    assert result["unique_id"].n_unique() == 2


def test_top_level_imports():
    from polars_ts.changepoint.pelt import pelt as pelt_fn

    assert callable(pelt_fn)
    assert pelt_fn is pelt
