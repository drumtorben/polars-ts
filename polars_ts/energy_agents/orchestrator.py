"""EnergyGridOrchestrator: hierarchical demand forecasting + reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from polars_ts.energy_agents.agents import (
    DemandForecastAgent,
    DemandResponseAgent,
    RenewableAgent,
    WeatherContextAgent,
)
from polars_ts.energy_agents.hierarchy import GridHierarchy
from polars_ts.reconciliation import reconcile


@dataclass
class EnergyForecastResult:
    """Output of an :class:`EnergyGridOrchestrator` run.

    Attributes
    ----------
    reconciled
        Long-format DataFrame of coherent per-node forecasts
        (``unique_id``, ``ds``, ``y_hat``).
    base_forecasts
        Per-node incoherent base forecasts before reconciliation.
    region_net_demand
        Region-level demand after subtracting renewable generation.
    demand_response
        Peak-shaving outcome: ``{"shifted_profile", "energy_shifted"}`` or
        ``None`` when no capacity limit was supplied.
    history
        Diagnostic records per node.

    """

    reconciled: pl.DataFrame
    base_forecasts: dict[str, np.ndarray]
    region_net_demand: np.ndarray
    demand_response: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class EnergyGridOrchestrator:
    """Hierarchical multi-agent energy/demand forecasting.

    Each node (household, grid, region) is forecast independently, optionally
    adjusted for weather; the incoherent forecasts are then reconciled to sum
    coherently across the hierarchy. Renewable generation and demand-response
    optimisation are applied at the region level.

    Parameters
    ----------
    season
        Seasonal period for the demand forecasters.
    reconcile_method
        Reconciliation method passed to
        :func:`polars_ts.reconciliation.reconcile`.

    """

    def __init__(self, season: int = 24, reconcile_method: str = "bottom_up") -> None:
        self.season = season
        self.reconcile_method = reconcile_method

    def run(
        self,
        household_histories: dict[str, np.ndarray],
        hierarchy: GridHierarchy,
        horizon: int,
        weather: np.ndarray | None = None,
        generation: np.ndarray | None = None,
        capacity: float | None = None,
    ) -> EnergyForecastResult:
        """Forecast, reconcile, and optimise demand across the grid hierarchy.

        Parameters
        ----------
        household_histories
            Mapping ``household_id -> 1D demand history``. Must cover every
            household in ``hierarchy``.
        hierarchy
            The region -> grid -> household topology.
        horizon
            Forecast horizon in steps.
        weather
            Optional region-wide temperature path of length ``horizon`` used to
            adjust every node's forecast.
        generation
            Optional region-level renewable generation path of length
            ``horizon`` subtracted from the reconciled region demand.
        capacity
            Optional region demand cap enabling peak-shaving demand response.

        Returns
        -------
        EnergyForecastResult

        """
        missing = set(hierarchy.households) - set(household_histories)
        if missing:
            raise ValueError(f"missing histories for households: {sorted(missing)}")

        forecaster = DemandForecastAgent(season=self.season)
        weather_agent = WeatherContextAgent()

        # Aggregate bottom-level histories up the tree so every node can forecast.
        node_history: dict[str, np.ndarray] = {
            h: np.asarray(household_histories[h], dtype=np.float64) for h in hierarchy.households
        }
        for grid, households in hierarchy.structure.items():
            node_history[grid] = np.sum([node_history[h] for h in households], axis=0)
        node_history[hierarchy.region] = np.sum([node_history[g] for g in hierarchy.grids], axis=0)

        base_forecasts: dict[str, np.ndarray] = {}
        history: list[dict[str, Any]] = []
        for node in hierarchy.all_nodes():
            fc = forecaster.forecast(node_history[node], horizon)
            if weather is not None:
                fc = weather_agent.adjust(fc, weather)
            base_forecasts[node] = fc
            history.append({"node": node, "mean_forecast": float(fc.mean())})

        # Bottom-up reconciliation aggregates the household (bottom-level)
        # forecasts up the tree, so only those are fed to reconcile; the
        # coherent grid/region series are derived from them.
        frames = [
            pl.DataFrame(
                {
                    "unique_id": [node] * horizon,
                    "ds": list(range(horizon)),
                    "y_hat": base_forecasts[node],
                }
            )
            for node in hierarchy.households
        ]
        df = pl.concat(frames)
        reconciled = reconcile(df, hierarchy.tree(), method=self.reconcile_method)

        region_demand = _node_series(reconciled, hierarchy.region, horizon)
        if generation is not None:
            region_demand = RenewableAgent().net_demand(region_demand, generation)

        demand_response = None
        if capacity is not None:
            shifted, energy = DemandResponseAgent(capacity=capacity).optimize(region_demand)
            demand_response = {"shifted_profile": shifted, "energy_shifted": energy}

        return EnergyForecastResult(
            reconciled=reconciled,
            base_forecasts=base_forecasts,
            region_net_demand=region_demand,
            demand_response=demand_response,
            history=history,
        )


def _node_series(reconciled: pl.DataFrame, node: str, horizon: int) -> np.ndarray:
    """Extract a node's reconciled forecast ordered by ``ds``."""
    sub = reconciled.filter(pl.col("unique_id") == node).sort("ds")
    values = sub["y_hat"].to_numpy()
    if values.shape[0] != horizon:
        raise ValueError(f"expected {horizon} reconciled points for {node!r}, got {values.shape[0]}")
    return values
