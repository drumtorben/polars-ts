"""Tests for hierarchical energy/demand forecasting agents (#162).

Defines the expected API for GridHierarchy, the demand/weather/renewable/
response agents, and the reconciliation-integrated EnergyGridOrchestrator.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hierarchy():
    from polars_ts.energy_agents import GridHierarchy

    return GridHierarchy("region", {"grid_a": ["h1", "h2"], "grid_b": ["h3"]})


@pytest.fixture
def histories() -> dict[str, np.ndarray]:
    """Build daily-seasonal household demand histories (season = 24)."""
    rng = np.random.default_rng(0)
    season = 24
    cycles = 5
    out = {}
    for i, h in enumerate(["h1", "h2", "h3"]):
        base = (1.0 + 0.3 * i) * (1.0 + np.sin(np.linspace(0, 2 * np.pi, season)))
        out[h] = np.tile(base, cycles) + rng.normal(0, 0.05, season * cycles)
    return out


# ---------------------------------------------------------------------------
# GridHierarchy
# ---------------------------------------------------------------------------


class TestGridHierarchy:
    def test_nodes(self, hierarchy):
        assert hierarchy.grids == ["grid_a", "grid_b"]
        assert hierarchy.households == ["h1", "h2", "h3"]
        assert hierarchy.all_nodes() == ["region", "grid_a", "grid_b", "h1", "h2", "h3"]

    def test_tree_mapping(self, hierarchy):
        tree = hierarchy.tree()
        assert tree["h1"] == "grid_a"
        assert tree["h3"] == "grid_b"
        assert tree["grid_a"] == "region"
        assert "region" not in tree  # top node has no parent

    def test_children(self, hierarchy):
        assert hierarchy.children("region") == ["grid_a", "grid_b"]
        assert hierarchy.children("grid_a") == ["h1", "h2"]
        assert hierarchy.children("h1") == []

    def test_empty_structure_raises(self):
        from polars_ts.energy_agents import GridHierarchy

        with pytest.raises(ValueError, match="at least one grid"):
            GridHierarchy("region", {})

    def test_duplicate_household_raises(self):
        from polars_ts.energy_agents import GridHierarchy

        with pytest.raises(ValueError, match="multiple grids"):
            GridHierarchy("region", {"g1": ["h1"], "g2": ["h1"]})


# ---------------------------------------------------------------------------
# DemandForecastAgent
# ---------------------------------------------------------------------------


class TestDemandForecastAgent:
    def test_seasonal_repeat(self):
        from polars_ts.energy_agents import DemandForecastAgent

        hist = np.tile(np.arange(24, dtype=float), 3)
        fc = DemandForecastAgent(season=24).forecast(hist, horizon=24)
        np.testing.assert_allclose(fc, np.arange(24))

    def test_horizon_longer_than_season(self):
        from polars_ts.energy_agents import DemandForecastAgent

        hist = np.tile(np.arange(24, dtype=float), 3)
        fc = DemandForecastAgent(season=24).forecast(hist, horizon=36)
        assert fc.shape == (36,)
        np.testing.assert_allclose(fc[:24], np.arange(24))

    def test_short_history_falls_back_to_mean(self):
        from polars_ts.energy_agents import DemandForecastAgent

        fc = DemandForecastAgent(season=24).forecast(np.array([2.0, 4.0]), horizon=5)
        np.testing.assert_allclose(fc, np.full(5, 3.0))

    def test_invalid_horizon(self):
        from polars_ts.energy_agents import DemandForecastAgent

        with pytest.raises(ValueError, match="horizon"):
            DemandForecastAgent().forecast(np.ones(30), horizon=0)


# ---------------------------------------------------------------------------
# WeatherContextAgent
# ---------------------------------------------------------------------------


class TestWeatherContextAgent:
    def test_comfort_no_change(self):
        from polars_ts.energy_agents import WeatherContextAgent

        agent = WeatherContextAgent(comfort_temp=18.0)
        base = np.ones(5)
        out = agent.adjust(base, np.full(5, 18.0))
        np.testing.assert_allclose(out, base)

    def test_heat_and_cold_raise_demand(self):
        from polars_ts.energy_agents import WeatherContextAgent

        agent = WeatherContextAgent(comfort_temp=18.0)
        base = np.ones(3)
        hot = agent.adjust(base, np.full(3, 30.0))
        cold = agent.adjust(base, np.full(3, 0.0))
        assert (hot > base).all()
        assert (cold > base).all()

    def test_shape_mismatch_raises(self):
        from polars_ts.energy_agents import WeatherContextAgent

        with pytest.raises(ValueError, match="horizon"):
            WeatherContextAgent().adjust(np.ones(5), np.ones(3))


# ---------------------------------------------------------------------------
# RenewableAgent
# ---------------------------------------------------------------------------


class TestRenewableAgent:
    def test_net_demand(self):
        from polars_ts.energy_agents import RenewableAgent

        net = RenewableAgent().net_demand(np.array([5.0, 3.0]), np.array([2.0, 4.0]))
        np.testing.assert_allclose(net, [3.0, -1.0])

    def test_curtail_floors_at_zero(self):
        from polars_ts.energy_agents import RenewableAgent

        net = RenewableAgent(curtail=True).net_demand(np.array([5.0, 3.0]), np.array([2.0, 4.0]))
        np.testing.assert_allclose(net, [3.0, 0.0])


# ---------------------------------------------------------------------------
# DemandResponseAgent
# ---------------------------------------------------------------------------


class TestDemandResponseAgent:
    def test_conserves_energy(self):
        from polars_ts.energy_agents import DemandResponseAgent

        profile = np.array([10.0, 2.0, 8.0, 1.0])
        shifted, energy = DemandResponseAgent(capacity=5.0).optimize(profile)
        assert shifted.sum() == pytest.approx(profile.sum())
        assert energy > 0

    def test_peaks_shaved(self):
        from polars_ts.energy_agents import DemandResponseAgent

        # Feasible window (total 21 <= capacity 6 * 4 slots): peaks fit under cap.
        profile = np.array([10.0, 2.0, 8.0, 1.0])
        shifted, energy = DemandResponseAgent(capacity=6.0).optimize(profile)
        assert shifted.max() <= 6.0 + 1e-9
        assert shifted.sum() == pytest.approx(profile.sum())
        assert energy > 0

    def test_infeasible_window_flattens(self):
        from polars_ts.energy_agents import DemandResponseAgent

        # total 21 > capacity 5 * 4 slots -> cannot fit under cap; flatten to mean.
        profile = np.array([10.0, 2.0, 8.0, 1.0])
        shifted, _ = DemandResponseAgent(capacity=5.0).optimize(profile)
        assert shifted.sum() == pytest.approx(profile.sum())
        np.testing.assert_allclose(shifted, np.full(4, profile.sum() / 4))

    def test_no_shift_when_under_capacity(self):
        from polars_ts.energy_agents import DemandResponseAgent

        profile = np.array([1.0, 2.0, 3.0])
        shifted, energy = DemandResponseAgent(capacity=5.0).optimize(profile)
        assert energy == 0.0
        np.testing.assert_allclose(shifted, profile)

    def test_invalid_capacity(self):
        from polars_ts.energy_agents import DemandResponseAgent

        with pytest.raises(ValueError, match="capacity"):
            DemandResponseAgent(capacity=0.0)


# ---------------------------------------------------------------------------
# EnergyGridOrchestrator
# ---------------------------------------------------------------------------


class TestEnergyGridOrchestrator:
    def test_run_returns_result(self, hierarchy, histories):
        from polars_ts.energy_agents import EnergyForecastResult, EnergyGridOrchestrator

        result = EnergyGridOrchestrator(season=24).run(histories, hierarchy, horizon=24)
        assert isinstance(result, EnergyForecastResult)
        assert isinstance(result.reconciled, pl.DataFrame)

    def test_forecasts_are_coherent(self, hierarchy, histories):
        from polars_ts.energy_agents import EnergyGridOrchestrator

        result = EnergyGridOrchestrator(season=24).run(histories, hierarchy, horizon=24)
        rec = result.reconciled

        def series(node):
            return rec.filter(pl.col("unique_id") == node).sort("ds")["y_hat"].to_numpy()

        # Household forecasts must sum to their grid; grids to the region.
        np.testing.assert_allclose(series("grid_a"), series("h1") + series("h2"), rtol=1e-6)
        np.testing.assert_allclose(series("region"), series("grid_a") + series("grid_b"), rtol=1e-6)

    def test_missing_history_raises(self, hierarchy):
        from polars_ts.energy_agents import EnergyGridOrchestrator

        with pytest.raises(ValueError, match="missing histories"):
            EnergyGridOrchestrator().run({"h1": np.ones(48)}, hierarchy, horizon=12)

    def test_weather_raises_demand(self, hierarchy, histories):
        from polars_ts.energy_agents import EnergyGridOrchestrator

        orch = EnergyGridOrchestrator(season=24)
        baseline = orch.run(histories, hierarchy, horizon=24)
        hot = orch.run(histories, hierarchy, horizon=24, weather=np.full(24, 35.0))
        assert hot.region_net_demand.sum() > baseline.region_net_demand.sum()

    def test_renewable_reduces_net_demand(self, hierarchy, histories):
        from polars_ts.energy_agents import EnergyGridOrchestrator

        orch = EnergyGridOrchestrator(season=24)
        base = orch.run(histories, hierarchy, horizon=24)
        gen = np.full(24, 1.0)
        with_gen = orch.run(histories, hierarchy, horizon=24, generation=gen)
        assert with_gen.region_net_demand.sum() < base.region_net_demand.sum()

    def test_demand_response_applied(self, hierarchy, histories):
        from polars_ts.energy_agents import EnergyGridOrchestrator

        orch = EnergyGridOrchestrator(season=24)
        result = orch.run(histories, hierarchy, horizon=24, capacity=0.1)
        assert result.demand_response is not None
        assert result.demand_response["energy_shifted"] >= 0.0


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    NAMES = [
        "GridHierarchy",
        "DemandForecastAgent",
        "WeatherContextAgent",
        "RenewableAgent",
        "DemandResponseAgent",
        "EnergyGridOrchestrator",
        "EnergyForecastResult",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_module(self, name):
        import polars_ts.energy_agents as mod

        assert getattr(mod, name) is not None
        assert name in mod.__all__

    @pytest.mark.parametrize("name", NAMES)
    def test_importable_from_top_level(self, name):
        import polars_ts

        assert getattr(polars_ts, name) is not None
