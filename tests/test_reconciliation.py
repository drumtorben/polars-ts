"""Tests for forecast reconciliation (#55)."""

from datetime import date

import polars as pl
import pytest

from polars_ts.reconciliation import reconcile


def _make_hierarchy() -> dict[str, str]:
    return {"A": "X", "B": "X", "C": "Y", "D": "Y", "X": "Total", "Y": "Total"}


def _make_forecasts() -> pl.DataFrame:
    """Incoherent forecasts at all levels."""
    return pl.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "X", "Y", "Total"] * 2,
            "ds": [date(2024, 1, 1)] * 7 + [date(2024, 1, 2)] * 7,
            "y_hat": [10.0, 20.0, 30.0, 40.0, 35.0, 65.0, 90.0] * 2,
        }
    )


class TestReconcile:
    def test_bottom_up(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        # Bottom-level values preserved, mid/top levels re-aggregated
        a = result.filter((pl.col("unique_id") == "A") & (pl.col("ds") == date(2024, 1, 1)))
        assert a["y_hat"][0] == pytest.approx(10.0)

    def test_bottom_up_mid_level(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        x = result.filter((pl.col("unique_id") == "X") & (pl.col("ds") == date(2024, 1, 1)))
        assert x["y_hat"][0] == pytest.approx(30.0)  # A + B

    def test_top_down(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="top_down")
        # Should produce forecasts for all levels
        ids = result["unique_id"].unique().to_list()
        assert "Total" in ids
        assert "A" in ids

    def test_ols(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="ols")
        # OLS should produce coherent forecasts
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        a = t1.filter(pl.col("unique_id") == "A")["y_hat"][0]
        b = t1.filter(pl.col("unique_id") == "B")["y_hat"][0]
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        assert x == pytest.approx(a + b, abs=0.1)

    def test_ols_coherent_total(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="ols")
        t1 = result.filter(pl.col("ds") == date(2024, 1, 1))
        x = t1.filter(pl.col("unique_id") == "X")["y_hat"][0]
        y = t1.filter(pl.col("unique_id") == "Y")["y_hat"][0]
        total = t1.filter(pl.col("unique_id") == "Total")["y_hat"][0]
        assert total == pytest.approx(x + y, abs=0.1)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            reconcile(_make_forecasts(), _make_hierarchy(), method="invalid")

    def test_multiple_timestamps(self):
        result = reconcile(_make_forecasts(), _make_hierarchy(), method="bottom_up")
        assert len(result["ds"].unique()) == 2


def test_top_level_import():
    import polars_ts

    assert polars_ts.reconcile is reconcile
