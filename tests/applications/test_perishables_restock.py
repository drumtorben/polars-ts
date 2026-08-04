"""Tests for the restock calculator (#211)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from polars_ts.applications.perishables import RestockPolicy, demand_std, recommend_orders


def _forecast(uid: str, daily: float, h: int = 14) -> pl.DataFrame:
    start = date(2025, 6, 1)
    return pl.DataFrame(
        {"unique_id": [uid] * h, "ds": [start + timedelta(days=i) for i in range(h)], "y_hat": [daily] * h}
    )


class TestRestockPolicy:
    def test_protection_days(self):
        policy = RestockPolicy(lead_time_days=2, review_period_days=3)
        assert policy.protection_days == 5

    def test_z_score_95(self):
        assert RestockPolicy(service_level=0.95).z_score == pytest.approx(1.6449, abs=1e-3)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="service_level"):
            RestockPolicy(service_level=1.5)
        with pytest.raises(ValueError, match="shelf_life_days"):
            RestockPolicy(shelf_life_days=0)
        with pytest.raises(ValueError, match="pack_size"):
            RestockPolicy(pack_size=0)


class TestRecommendOrders:
    def test_no_uncertainty_no_stock(self):
        # 10/day, protection = 3 days -> order 30
        policy = RestockPolicy(lead_time_days=2, review_period_days=1)
        out = recommend_orders(_forecast("A", 10.0), policy)
        assert out["forecast_demand"].item() == pytest.approx(30.0)
        assert out["safety_stock"].item() == 0.0
        assert out["order_qty"].item() == 30

    def test_on_hand_netted(self):
        policy = RestockPolicy(lead_time_days=2, review_period_days=1)
        on_hand = pl.DataFrame({"unique_id": ["A"], "on_hand": [12.0]})
        out = recommend_orders(_forecast("A", 10.0), policy, on_hand=on_hand)
        assert out["order_qty"].item() == 18

    def test_safety_stock_scales_with_sigma(self):
        policy = RestockPolicy(service_level=0.95, lead_time_days=3, review_period_days=1)
        sigma = pl.DataFrame({"unique_id": ["A"], "sigma": [5.0]})
        out = recommend_orders(_forecast("A", 10.0), policy, sigma=sigma)
        expected_ss = policy.z_score * 5.0 * 2.0  # sqrt(4 protection days)
        assert out["safety_stock"].item() == pytest.approx(expected_ss)
        assert out["order_qty"].item() == pytest.approx(40 + expected_ss, abs=1.0)

    def test_shelf_life_caps_order(self):
        # Protection period wants 7 days of demand, but shelf life allows only 2 sellable days
        policy = RestockPolicy(lead_time_days=0, review_period_days=7, shelf_life_days=2)
        out = recommend_orders(_forecast("A", 10.0), policy)
        assert out["order_qty"].item() == 20

    def test_shelf_life_cap_accounts_on_hand(self):
        policy = RestockPolicy(lead_time_days=0, review_period_days=7, shelf_life_days=2)
        on_hand = pl.DataFrame({"unique_id": ["A"], "on_hand": [15.0]})
        out = recommend_orders(_forecast("A", 10.0), policy, on_hand=on_hand)
        assert out["order_qty"].item() == 5

    def test_moq_and_pack_size(self):
        policy = RestockPolicy(lead_time_days=0, review_period_days=1, moq=10, pack_size=6)
        out = recommend_orders(_forecast("A", 3.0), policy)
        # raw order 3 -> raised to moq 10 -> rounded up to 12
        assert out["order_qty"].item() == 12

    def test_zero_order_stays_zero(self):
        policy = RestockPolicy(lead_time_days=0, review_period_days=1, moq=10)
        on_hand = pl.DataFrame({"unique_id": ["A"], "on_hand": [100.0]})
        out = recommend_orders(_forecast("A", 3.0), policy, on_hand=on_hand)
        assert out["order_qty"].item() == 0

    def test_short_horizon_raises(self):
        policy = RestockPolicy(lead_time_days=10, review_period_days=10)
        with pytest.raises(ValueError, match="protection period"):
            recommend_orders(_forecast("A", 10.0, h=5), policy)

    def test_multiple_skus(self):
        fc = pl.concat([_forecast("A", 10.0), _forecast("B", 1.0)])
        out = recommend_orders(fc, RestockPolicy(lead_time_days=1, review_period_days=1))
        assert out["unique_id"].to_list() == ["A", "B"]
        assert out["order_qty"].to_list() == [20, 2]


class TestDemandStd:
    def test_windowed_std(self):
        start = date(2025, 1, 1)
        n = 200
        # Old history is wild, recent 91 days are constant
        y = [100.0, 0.0] * 55 + [5.0] * 90
        df = pl.DataFrame({"unique_id": ["A"] * n, "ds": [start + timedelta(days=i) for i in range(n)], "y": y})
        full = demand_std(df, window=None)["sigma"].item()
        recent = demand_std(df, window=91)["sigma"].item()
        assert recent < 2.0 < full

    def test_single_row_zero(self):
        df = pl.DataFrame({"unique_id": ["A"], "ds": [date(2025, 1, 1)], "y": [5.0]})
        assert demand_std(df)["sigma"].item() == 0.0
