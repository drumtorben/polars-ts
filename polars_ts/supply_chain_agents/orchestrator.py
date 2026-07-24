"""SupplyChainOrchestrator: demand sensing -> inventory -> echelon coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polars_ts.supply_chain_agents.agents import (
    DemandSensingAgent,
    EchelonCoordinatorAgent,
    InventoryAgent,
    PromotionEffectAgent,
)


@dataclass
class SupplyChainResult:
    """Output of a :class:`SupplyChainOrchestrator` run.

    Attributes
    ----------
    sensed_demand
        Signal- and promotion-adjusted demand forecast.
    promo_lift
        Estimated multiplicative promotion lift.
    reorder
        Inventory reorder decision (see :meth:`InventoryAgent.reorder`).
    echelon_orders
        Order series at each echelon (index 0 = store demand).
    bullwhip_ratio
        Order-variance amplification from bottom to top echelon.
    history
        Diagnostic records.

    """

    sensed_demand: np.ndarray
    promo_lift: float
    reorder: dict[str, float]
    echelon_orders: list[np.ndarray]
    bullwhip_ratio: float
    history: list[dict[str, Any]] = field(default_factory=list)


class SupplyChainOrchestrator:
    """Coordinate demand-sensing, inventory, and multi-echelon agents.

    Pipeline: build a seasonal baseline from POS history (or accept a
    foundation-model ``base_forecast``), estimate and apply promotion lift,
    fuse external signals into sensed demand, derive an inventory reorder
    decision, and propagate orders up the supply chain.

    Parameters
    ----------
    season
        Seasonal period for the POS baseline forecaster.
    lead_time
        Replenishment lead time for the inventory policy.
    n_echelons
        Number of upstream echelons to coordinate.

    """

    def __init__(self, season: int = 7, lead_time: int = 2, n_echelons: int = 2) -> None:
        self.season = season
        self.lead_time = lead_time
        self.n_echelons = n_echelons

    def _baseline(self, pos_history: np.ndarray, horizon: int) -> np.ndarray:
        """Seasonal-naive POS baseline (mean fallback for short history)."""
        h = np.asarray(pos_history, dtype=np.float64)
        if h.size < self.season:
            return np.full(horizon, float(h.mean()) if h.size else 0.0)
        cycle = h[-self.season :]
        reps = int(np.ceil(horizon / self.season))
        return np.tile(cycle, reps)[:horizon]

    def run(
        self,
        pos_history: np.ndarray,
        horizon: int,
        signals: dict[str, np.ndarray] | None = None,
        signal_weights: dict[str, float] | None = None,
        promo_history: tuple[np.ndarray, np.ndarray] | None = None,
        promo_schedule: np.ndarray | None = None,
        on_hand: float = 0.0,
        base_forecast: np.ndarray | None = None,
    ) -> SupplyChainResult:
        """Run the full demand-sensing and coordination pipeline.

        Parameters
        ----------
        pos_history
            Historical point-of-sale demand.
        horizon
            Forecast horizon in steps.
        signals
            Optional external signal uplift paths (social, weather, events).
        signal_weights
            Optional per-signal fusion weights.
        promo_history
            Optional ``(sales, promo_flags)`` for lift estimation.
        promo_schedule
            Optional future promotion indicator path (length ``horizon``).
        on_hand
            Current on-hand inventory.
        base_forecast
            Optional externally supplied baseline (e.g. a foundation-model
            forecast) used instead of the seasonal POS baseline.

        Returns
        -------
        SupplyChainResult

        """
        baseline = (
            np.asarray(base_forecast, dtype=np.float64)
            if base_forecast is not None
            else self._baseline(pos_history, horizon)
        )
        if baseline.shape[0] != horizon:
            raise ValueError(f"baseline length {baseline.shape[0]} != horizon {horizon}")

        # Promotion effect.
        promo_agent = PromotionEffectAgent()
        lift = 0.0
        if promo_history is not None:
            lift = promo_agent.estimate(*promo_history)
        if promo_schedule is not None:
            baseline = promo_agent.apply(baseline, promo_schedule, lift)

        # Multi-source demand sensing.
        sensed = DemandSensingAgent(weights=signal_weights).sense(baseline, signals)

        # Inventory policy.
        reorder = InventoryAgent(lead_time=self.lead_time).reorder(sensed, on_hand)

        # Multi-echelon coordination.
        coord = EchelonCoordinatorAgent(n_echelons=self.n_echelons).coordinate(sensed)
        echelon_orders: list[np.ndarray] = coord["echelon_orders"]
        bullwhip = float(coord["bullwhip_ratio"])

        history = [
            {"stage": "baseline", "mean": float(np.mean(self._safe(base_forecast, pos_history)))},
            {"stage": "promo_lift", "lift": lift},
            {"stage": "sensed", "mean": float(sensed.mean())},
            {"stage": "reorder", "order_qty": reorder["order_qty"]},
            {"stage": "bullwhip", "ratio": bullwhip},
        ]

        return SupplyChainResult(
            sensed_demand=sensed,
            promo_lift=lift,
            reorder=reorder,
            echelon_orders=echelon_orders,
            bullwhip_ratio=bullwhip,
            history=history,
        )

    @staticmethod
    def _safe(base_forecast: np.ndarray | None, pos_history: np.ndarray) -> np.ndarray:
        """Return a non-empty array for diagnostic means."""
        if base_forecast is not None:
            return np.asarray(base_forecast, dtype=np.float64)
        h = np.asarray(pos_history, dtype=np.float64)
        return h if h.size else np.zeros(1)
