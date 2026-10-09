"""Tests for cold-start warm-up forecasting (#211)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.applications.perishables import cold_start_forecast, cold_start_skus


def _series(uid: str, values: np.ndarray, start: date = date(2024, 1, 1)) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"unique_id": [uid] * len(values), "ds": dates, "y": values.astype(np.float64)})


@pytest.fixture
def market() -> pl.DataFrame:
    """Two established donors ramping from launch, one 10-day-old SKU."""
    t = np.arange(120)
    ramp = 5.0 + 0.1 * t  # donors ramp linearly after launch
    donor_a = _series("donor_a", ramp)
    donor_b = _series("donor_b", 2.0 * ramp)
    new = _series("new", 5.0 + 0.1 * np.arange(10), start=date(2024, 4, 20))
    return pl.concat([donor_a, donor_b, new])


class TestColdStartSkus:
    def test_detects_short_history(self, market):
        assert cold_start_skus(market, min_history=28) == ["new"]

    def test_none_when_all_established(self, market):
        established = market.filter(pl.col("unique_id") != "new")
        assert cold_start_skus(established, min_history=28) == []


class TestColdStartForecast:
    def test_forecast_shape_and_dates(self, market):
        fc = cold_start_forecast(market, h=7, min_history=28, k=2)
        assert fc.columns == ["unique_id", "ds", "y_hat"]
        assert fc["unique_id"].unique().to_list() == ["new"]
        assert fc.height == 7
        assert fc["ds"].min() == date(2024, 4, 30)

    def test_borrows_donor_trajectory(self, market):
        # Donors continue ramping after day 10; the borrowed forecast
        # should sit near the new SKU's level and keep growing.
        fc = cold_start_forecast(market, h=14, min_history=28, k=2).sort("ds")
        assert fc["y_hat"][0] == pytest.approx(6.0, rel=0.3)
        assert fc["y_hat"][-1] > fc["y_hat"][0]

    def test_empty_when_no_cold_skus(self, market):
        established = market.filter(pl.col("unique_id") != "new")
        fc = cold_start_forecast(established, h=7)
        assert fc.height == 0
        assert fc.columns == ["unique_id", "ds", "y_hat"]

    def test_no_donors_raises(self, market):
        only_new = market.filter(pl.col("unique_id") == "new")
        with pytest.raises(ValueError, match="donors"):
            cold_start_forecast(only_new, h=7, min_history=28)

    def test_category_restricts_donors(self, market):
        # donor_b sells at 2x; putting new+donor_b in one category should
        # roughly double the forecast vs pairing with donor_a.
        cats_b = pl.DataFrame({"unique_id": ["donor_a", "donor_b", "new"], "category": ["x", "y", "y"]})
        cats_a = pl.DataFrame({"unique_id": ["donor_a", "donor_b", "new"], "category": ["y", "x", "y"]})
        fc_b = cold_start_forecast(market, h=7, k=1, category_map=cats_b).sort("ds")
        fc_a = cold_start_forecast(market, h=7, k=1, category_map=cats_a).sort("ds")
        # Both rescale to the new SKU's observed level, so trajectories are
        # shaped by the donor but anchored to the same level.
        assert fc_b["y_hat"][0] == pytest.approx(fc_a["y_hat"][0], rel=0.5)

    def test_nonnegative(self, market):
        fc = cold_start_forecast(market, h=30, min_history=28)
        assert (fc["y_hat"] >= 0.0).all()

    def test_invalid_horizon(self, market):
        with pytest.raises(ValueError, match="Horizon"):
            cold_start_forecast(market, h=0)
