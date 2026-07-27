"""Agents for supply-chain demand sensing and inventory coordination.

- :class:`DemandSensingAgent` — fuse POS, social, weather, and event signals.
- :class:`PromotionEffectAgent` — estimate and apply promotion lift.
- :class:`InventoryAgent` — inventory-aware base-stock reorder policy.
- :class:`EchelonCoordinatorAgent` — multi-echelon order propagation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DemandSensingAgent:
    """Fuse a baseline demand forecast with external demand signals.

    Each signal is a fractional uplift path (e.g. a normalised social-buzz or
    event index); the sensed demand is ``baseline * (1 + sum_i w_i * signal_i)``
    floored at zero. This is the multi-source fusion step (POS baseline +
    social / weather / events), and ``baseline`` may originate from a
    foundation-model forecast for context-aware sensing.

    Parameters
    ----------
    weights
        Mapping ``signal_name -> weight``. Signals absent from this mapping
        default to weight ``1.0``.

    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def sense(self, baseline: np.ndarray, signals: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Return the signal-fused sensed demand for ``baseline``."""
        base = np.asarray(baseline, dtype=np.float64)
        if not signals:
            return base.copy()
        uplift = np.zeros_like(base)
        for name, sig in signals.items():
            sig = np.asarray(sig, dtype=np.float64)
            if sig.shape != base.shape:
                raise ValueError(f"signal {name!r} must match the forecast horizon")
            uplift += self.weights.get(name, 1.0) * sig
        return np.clip(base * (1.0 + uplift), 0.0, None)


class PromotionEffectAgent:
    """Estimate multiplicative promotion lift and apply it to a forecast.

    The lift is a difference-in-means estimate — mean sales on promoted periods
    versus non-promoted — a lightweight causal-style contrast (see the causal
    inference module, T1-5, for confounder-adjusted alternatives).
    """

    def estimate(self, sales: np.ndarray, promo_flags: np.ndarray) -> float:
        """Return the multiplicative lift ``(promo_mean / base_mean) - 1``."""
        sales = np.asarray(sales, dtype=np.float64)
        promo = np.asarray(promo_flags, dtype=bool)
        if sales.shape != promo.shape:
            raise ValueError("sales and promo_flags must have the same length")
        if not promo.any() or promo.all():
            return 0.0
        base_mean = float(sales[~promo].mean())
        promo_mean = float(sales[promo].mean())
        if base_mean <= 0.0:
            return 0.0
        return promo_mean / base_mean - 1.0

    def apply(self, forecast: np.ndarray, promo_schedule: np.ndarray, lift: float) -> np.ndarray:
        """Scale ``forecast`` by ``lift`` on periods flagged in ``promo_schedule``."""
        forecast = np.asarray(forecast, dtype=np.float64)
        schedule = np.asarray(promo_schedule, dtype=np.float64)
        if schedule.shape != forecast.shape:
            raise ValueError("promo_schedule must match the forecast horizon")
        return forecast * (1.0 + lift * schedule)


class InventoryAgent:
    """Inventory-aware base-stock (order-up-to) reorder policy.

    Parameters
    ----------
    lead_time
        Replenishment lead time in steps.
    safety_factor
        Service-level multiplier (``z``) applied to lead-time demand std.

    """

    def __init__(self, lead_time: int = 1, safety_factor: float = 1.65) -> None:
        if lead_time < 1:
            raise ValueError("lead_time must be >= 1")
        self.lead_time = lead_time
        self.safety_factor = safety_factor

    def reorder(self, demand_forecast: np.ndarray, on_hand: float) -> dict[str, float]:
        """Return the reorder decision for the coming lead-time window.

        Returns
        -------
        dict
            ``order_up_to``, ``safety_stock``, ``order_qty`` and a boolean
            ``stockout_risk`` (as ``0.0`` / ``1.0``).

        """
        f = np.asarray(demand_forecast, dtype=np.float64)
        window = f[: self.lead_time]
        lead_demand = float(window.sum())
        std = float(f.std()) if f.size > 1 else 0.0
        safety_stock = self.safety_factor * std * np.sqrt(self.lead_time)
        order_up_to = lead_demand + safety_stock
        order_qty = max(order_up_to - on_hand, 0.0)
        return {
            "order_up_to": order_up_to,
            "safety_stock": float(safety_stock),
            "order_qty": float(order_qty),
            "stockout_risk": 1.0 if on_hand < lead_demand else 0.0,
        }


class EchelonCoordinatorAgent:
    """Propagate orders up a multi-echelon chain (store -> DC -> factory).

    Each echelon smooths downstream orders with an exponential filter; the
    coordinator also reports the bullwhip ratio (order variance amplification
    from the bottom to the top echelon).

    Parameters
    ----------
    n_echelons
        Number of echelons above the demand source.
    smoothing
        Exponential smoothing factor in ``[0, 1]`` applied at each echelon
        (1 = pass-through, lower = more smoothing / less bullwhip).

    """

    def __init__(self, n_echelons: int = 2, smoothing: float = 0.5) -> None:
        if n_echelons < 1:
            raise ValueError("n_echelons must be >= 1")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.n_echelons = n_echelons
        self.smoothing = smoothing

    def _smooth(self, series: np.ndarray) -> np.ndarray:
        out = np.empty_like(series)
        level = float(series[0])
        for i, v in enumerate(series):
            level = self.smoothing * float(v) + (1.0 - self.smoothing) * level
            out[i] = level
        return out

    def coordinate(self, demand: np.ndarray) -> dict[str, Any]:
        """Return per-echelon order series and the bullwhip ratio."""
        demand = np.asarray(demand, dtype=np.float64)
        orders = [demand]
        for _ in range(self.n_echelons):
            orders.append(self._smooth(orders[-1]))
        base_var = float(np.var(demand)) + 1e-12
        bullwhip = float(np.var(orders[-1]) / base_var)
        return {"echelon_orders": orders, "bullwhip_ratio": bullwhip}
