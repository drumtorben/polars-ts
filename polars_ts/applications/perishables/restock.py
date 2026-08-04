"""Forecast-to-order translation for perishables restocking.

The deliverable of the application: "how many units of SKU X to order
for the next N days". Implements an order-up-to policy with a
service-level safety stock, netted against current inventory, and — the
perishables twist — a shelf-life cap so orders never exceed what can
sell before spoiling. Over-forecast is waste, under-forecast is
stockouts; the policy makes that trade-off explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import polars as pl


@dataclass(frozen=True)
class RestockPolicy:
    """Ordering policy parameters for perishables replenishment.

    Parameters
    ----------
    service_level
        Target cycle service level (probability of no stockout during
        the protection period), e.g. 0.95.
    lead_time_days
        Days between placing an order and receiving it.
    review_period_days
        Days between consecutive ordering opportunities.
    shelf_life_days
        Sellable days after receipt; when set, orders are capped so
        total stock never exceeds forecast demand within shelf life.
    moq
        Minimum order quantity — nonzero orders are raised to this.
    pack_size
        Case/pack multiple — orders are rounded up to it.

    """

    service_level: float = 0.95
    lead_time_days: int = 1
    review_period_days: int = 1
    shelf_life_days: int | None = None
    moq: int = 0
    pack_size: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.service_level < 1.0:
            raise ValueError("service_level must be in (0, 1)")
        if self.lead_time_days < 0 or self.review_period_days < 1:
            raise ValueError("require lead_time_days >= 0 and review_period_days >= 1")
        if self.shelf_life_days is not None and self.shelf_life_days < 1:
            raise ValueError("shelf_life_days must be >= 1 when set")
        if self.moq < 0 or self.pack_size < 1:
            raise ValueError("require moq >= 0 and pack_size >= 1")

    @property
    def protection_days(self) -> int:
        """Days of demand an order must cover (lead time + review period)."""
        return self.lead_time_days + self.review_period_days

    @property
    def z_score(self) -> float:
        """Standard-normal quantile for the target service level."""
        return NormalDist().inv_cdf(self.service_level)


def demand_std(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    window: int | None = 91,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Estimate per-SKU daily demand standard deviation.

    A fallback uncertainty estimate when backtest forecast-error
    residuals are not available; computed over each SKU's trailing
    window of history.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    target_col, id_col, time_col
        Canonical column names.
    window
        Trailing days to use; ``None`` uses full history.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "sigma"]``.

    """
    out = df.sort(id_col, time_col)
    if window is not None:
        age = (pl.col(time_col).max().over(id_col) - pl.col(time_col)).dt.total_days()
        out = out.filter(age < window)
    return (
        out.group_by(id_col)
        .agg(pl.col(target_col).std().fill_null(0.0).alias("sigma"))
        .with_columns(pl.col("sigma").fill_null(0.0))
        .sort(id_col)
    )


def recommend_orders(
    forecast: pl.DataFrame,
    policy: RestockPolicy | None = None,
    on_hand: pl.DataFrame | None = None,
    sigma: pl.DataFrame | None = None,
    target_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Translate per-day forecasts into order quantities per SKU.

    Implements an order-up-to-S policy: the base stock covers forecast
    demand over the protection period (lead time + review period) plus a
    safety stock ``z * sigma * sqrt(protection_days)`` for the target
    service level. The order is the gap between S and current inventory,
    capped at forecast demand within shelf life (minus stock already on
    hand) so perishables are not ordered into spoilage, then raised to
    the MOQ and rounded up to the pack size.

    Parameters
    ----------
    forecast
        Per-day forecast frame ``[id_col, time_col, target_col]``
        starting the day after the order is placed; must cover at least
        the protection period (and shelf life, when capping is enabled).
    policy
        Restock policy; defaults to ``RestockPolicy()``.
    on_hand
        Optional ``[id_col, "on_hand"]`` current sellable inventory
        (units on order may be included here); missing SKUs default 0.
    sigma
        Optional ``[id_col, "sigma"]`` daily forecast-error standard
        deviation (see :func:`demand_std`); missing SKUs default 0,
        i.e. no safety stock.
    target_col, id_col, time_col
        Column names in ``forecast``.

    Returns
    -------
    pl.DataFrame
        One row per SKU with columns ``[id_col, "forecast_demand",
        "safety_stock", "order_up_to", "on_hand", "shelf_life_cap",
        "order_qty"]``.

    """
    policy = policy or RestockPolicy()
    horizon = forecast.group_by(id_col).agg(pl.len().alias("__h"))["__h"].min()
    if horizon is not None and horizon < policy.protection_days:
        raise ValueError(f"Forecast horizon ({horizon}) shorter than protection period ({policy.protection_days})")

    sorted_fc = forecast.sort(id_col, time_col).with_columns((pl.int_range(pl.len()).over(id_col) + 1).alias("__step"))
    demand_pp = pl.col(target_col).filter(pl.col("__step") <= policy.protection_days).sum()
    if policy.shelf_life_days is not None:
        shelf_horizon = policy.lead_time_days + policy.shelf_life_days
        shelf_expr = pl.col(target_col).filter(pl.col("__step") <= shelf_horizon).sum()
    else:
        shelf_expr = pl.lit(None, dtype=pl.Float64)

    out = sorted_fc.group_by(id_col).agg(
        demand_pp.alias("forecast_demand"),
        shelf_expr.alias("__shelf_demand"),
    )

    sigma_df = sigma if sigma is not None else out.select(pl.col(id_col), pl.lit(0.0).alias("sigma"))
    on_hand_df = on_hand if on_hand is not None else out.select(pl.col(id_col), pl.lit(0.0).alias("on_hand"))
    out = (
        out.join(sigma_df, on=id_col, how="left")
        .join(on_hand_df, on=id_col, how="left")
        .with_columns(pl.col("sigma").fill_null(0.0), pl.col("on_hand").fill_null(0.0).cast(pl.Float64))
    )

    sqrt_pp = math.sqrt(policy.protection_days)
    out = out.with_columns(
        (policy.z_score * pl.col("sigma") * sqrt_pp).alias("safety_stock"),
    ).with_columns(
        (pl.col("forecast_demand") + pl.col("safety_stock")).alias("order_up_to"),
        (pl.col("__shelf_demand") - pl.col("on_hand")).clip(lower_bound=0.0).alias("shelf_life_cap"),
    )

    raw_order = (pl.col("order_up_to") - pl.col("on_hand")).clip(lower_bound=0.0)
    capped = (
        pl.when(pl.col("shelf_life_cap").is_not_null())
        .then(pl.min_horizontal(raw_order, pl.col("shelf_life_cap")))
        .otherwise(raw_order)
    )
    rounded = (
        pl.when(capped <= 0)
        .then(0.0)
        .otherwise(
            ((pl.max_horizontal(capped, pl.lit(float(policy.moq))) / policy.pack_size).ceil()) * policy.pack_size
        )
    )
    return (
        out.with_columns(rounded.cast(pl.Int64).alias("order_qty"))
        .drop("sigma", "__shelf_demand")
        .select(id_col, "forecast_demand", "safety_stock", "order_up_to", "on_hand", "shelf_life_cap", "order_qty")
        .sort(id_col)
    )
