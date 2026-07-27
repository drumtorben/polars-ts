"""Tests for supply-chain demand-sensing agents (#163).

Defines the expected API for the sensing/promotion/inventory/echelon agents
and the SupplyChainOrchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pos_history() -> np.ndarray:
    """Build weekly-seasonal POS demand history."""
    rng = np.random.default_rng(0)
    week = np.array([100.0, 90.0, 95.0, 110.0, 130.0, 160.0, 140.0])
    return np.tile(week, 8) + rng.normal(0, 3, 56)


# ---------------------------------------------------------------------------
# DemandSensingAgent
# ---------------------------------------------------------------------------


class TestDemandSensingAgent:
    def test_no_signals_returns_baseline(self):
        from polars_ts.supply_chain_agents import DemandSensingAgent

        base = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(DemandSensingAgent().sense(base), base)

    def test_positive_signal_raises_demand(self):
        from polars_ts.supply_chain_agents import DemandSensingAgent

        base = np.array([10.0, 10.0])
        out = DemandSensingAgent().sense(base, {"social": np.array([0.5, 0.5])})
        np.testing.assert_allclose(out, [15.0, 15.0])

    def test_weighted_fusion(self):
        from polars_ts.supply_chain_agents import DemandSensingAgent

        base = np.array([10.0])
        agent = DemandSensingAgent(weights={"social": 2.0})
        out = agent.sense(base, {"social": np.array([0.5])})
        np.testing.assert_allclose(out, [20.0])  # 10 * (1 + 2*0.5)

    def test_negative_signal_floored(self):
        from polars_ts.supply_chain_agents import DemandSensingAgent

        base = np.array([10.0])
        out = DemandSensingAgent().sense(base, {"x": np.array([-5.0])})
        assert out[0] == 0.0

    def test_shape_mismatch_raises(self):
        from polars_ts.supply_chain_agents import DemandSensingAgent

        with pytest.raises(ValueError, match="horizon"):
            DemandSensingAgent().sense(np.ones(3), {"s": np.ones(2)})


# ---------------------------------------------------------------------------
# PromotionEffectAgent
# ---------------------------------------------------------------------------


class TestPromotionEffectAgent:
    def test_estimates_positive_lift(self):
        from polars_ts.supply_chain_agents import PromotionEffectAgent

        sales = np.array([100.0, 100.0, 150.0, 150.0])
        promo = np.array([False, False, True, True])
        lift = PromotionEffectAgent().estimate(sales, promo)
        assert lift == pytest.approx(0.5)

    def test_no_promo_zero_lift(self):
        from polars_ts.supply_chain_agents import PromotionEffectAgent

        sales = np.array([100.0, 120.0, 90.0])
        assert PromotionEffectAgent().estimate(sales, np.zeros(3, dtype=bool)) == 0.0

    def test_apply_scales_scheduled_periods(self):
        from polars_ts.supply_chain_agents import PromotionEffectAgent

        fc = np.array([100.0, 100.0])
        out = PromotionEffectAgent().apply(fc, np.array([0.0, 1.0]), lift=0.5)
        np.testing.assert_allclose(out, [100.0, 150.0])

    def test_mismatch_raises(self):
        from polars_ts.supply_chain_agents import PromotionEffectAgent

        with pytest.raises(ValueError, match="same length"):
            PromotionEffectAgent().estimate(np.ones(3), np.ones(2, dtype=bool))


# ---------------------------------------------------------------------------
# InventoryAgent
# ---------------------------------------------------------------------------


class TestInventoryAgent:
    def test_orders_when_short(self):
        from polars_ts.supply_chain_agents import InventoryAgent

        demand = np.full(10, 20.0)
        out = InventoryAgent(lead_time=3).reorder(demand, on_hand=0.0)
        assert out["order_qty"] > 0
        assert out["order_up_to"] >= 60.0  # 3 * 20 lead-time demand

    def test_no_order_when_well_stocked(self):
        from polars_ts.supply_chain_agents import InventoryAgent

        demand = np.full(10, 20.0)
        out = InventoryAgent(lead_time=2).reorder(demand, on_hand=10_000.0)
        assert out["order_qty"] == 0.0

    def test_stockout_risk_flag(self):
        from polars_ts.supply_chain_agents import InventoryAgent

        demand = np.full(10, 20.0)
        risky = InventoryAgent(lead_time=3).reorder(demand, on_hand=10.0)
        safe = InventoryAgent(lead_time=3).reorder(demand, on_hand=1000.0)
        assert risky["stockout_risk"] == 1.0
        assert safe["stockout_risk"] == 0.0

    def test_invalid_lead_time(self):
        from polars_ts.supply_chain_agents import InventoryAgent

        with pytest.raises(ValueError, match="lead_time"):
            InventoryAgent(lead_time=0)


# ---------------------------------------------------------------------------
# EchelonCoordinatorAgent
# ---------------------------------------------------------------------------


class TestEchelonCoordinatorAgent:
    def test_produces_orders_per_echelon(self):
        from polars_ts.supply_chain_agents import EchelonCoordinatorAgent

        demand = np.array([10.0, 20.0, 5.0, 30.0, 8.0])
        out = EchelonCoordinatorAgent(n_echelons=3).coordinate(demand)
        assert len(out["echelon_orders"]) == 4  # source + 3 echelons

    def test_smoothing_reduces_variance(self):
        from polars_ts.supply_chain_agents import EchelonCoordinatorAgent

        demand = np.array([10.0, 40.0, 5.0, 50.0, 8.0, 45.0])
        out = EchelonCoordinatorAgent(n_echelons=2, smoothing=0.3).coordinate(demand)
        # Smoothing dampens variance upstream -> bullwhip ratio below 1.
        assert out["bullwhip_ratio"] < 1.0

    def test_invalid_smoothing(self):
        from polars_ts.supply_chain_agents import EchelonCoordinatorAgent

        with pytest.raises(ValueError, match="smoothing"):
            EchelonCoordinatorAgent(smoothing=1.5)


# ---------------------------------------------------------------------------
# SupplyChainOrchestrator
# ---------------------------------------------------------------------------


class TestSupplyChainOrchestrator:
    def test_run_returns_result(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator, SupplyChainResult

        result = SupplyChainOrchestrator(season=7).run(pos_history, horizon=7)
        assert isinstance(result, SupplyChainResult)
        assert result.sensed_demand.shape == (7,)

    def test_signals_increase_sensed_demand(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        orch = SupplyChainOrchestrator(season=7)
        base = orch.run(pos_history, horizon=7)
        boosted = orch.run(pos_history, horizon=7, signals={"social": np.full(7, 0.2)})
        assert boosted.sensed_demand.sum() > base.sensed_demand.sum()

    def test_promotion_lift_applied(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        sales = np.array([100.0, 100.0, 200.0, 200.0])
        promo = np.array([False, False, True, True])
        result = SupplyChainOrchestrator(season=7).run(
            pos_history,
            horizon=7,
            promo_history=(sales, promo),
            promo_schedule=np.ones(7),
        )
        assert result.promo_lift == pytest.approx(1.0)

    def test_reorder_present(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        result = SupplyChainOrchestrator(season=7, lead_time=2).run(pos_history, horizon=7, on_hand=0.0)
        assert result.reorder["order_qty"] > 0

    def test_foundation_base_forecast_used(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        base = np.full(7, 500.0)  # e.g. a foundation-model forecast
        result = SupplyChainOrchestrator(season=7).run(pos_history, horizon=7, base_forecast=base)
        np.testing.assert_allclose(result.sensed_demand, base)

    def test_base_forecast_wrong_length_raises(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        with pytest.raises(ValueError, match="horizon"):
            SupplyChainOrchestrator().run(pos_history, horizon=7, base_forecast=np.ones(3))

    def test_echelon_orders_reported(self, pos_history):
        from polars_ts.supply_chain_agents import SupplyChainOrchestrator

        result = SupplyChainOrchestrator(season=7, n_echelons=3).run(pos_history, horizon=14)
        assert len(result.echelon_orders) == 4
        assert result.bullwhip_ratio >= 0.0


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    NAMES = [
        "DemandSensingAgent",
        "PromotionEffectAgent",
        "InventoryAgent",
        "EchelonCoordinatorAgent",
        "SupplyChainOrchestrator",
        "SupplyChainResult",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_module(self, name):
        import polars_ts.supply_chain_agents as mod

        assert getattr(mod, name) is not None
        assert name in mod.__all__

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_top_level(self, name):
        import polars_ts

        assert getattr(polars_ts, name) is not None
