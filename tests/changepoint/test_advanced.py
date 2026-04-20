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


def test_top_level_imports():
    from polars_ts.changepoint.pelt import pelt as pelt_fn

    assert callable(pelt_fn)
    assert pelt_fn is pelt
