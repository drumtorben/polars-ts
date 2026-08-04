"""Perishables retail demand forecasting application (#211).

End-to-end tooling for restocking decisions on perishable goods:
CSV ingestion with schema validation, growth-aware preprocessing,
intermittent-demand models (Croston/SBA/TSB), cold-start warm-up for
new SKUs, domain feature engineering, and a forecast-to-order
restock calculator with service-level and shelf-life constraints.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "ColumnMapping": ("polars_ts.applications.perishables.loader", "ColumnMapping"),
    "load_sales_csv": ("polars_ts.applications.perishables.loader", "load_sales_csv"),
    "validate_sales_frame": ("polars_ts.applications.perishables.loader", "validate_sales_frame"),
    "fill_missing_dates": ("polars_ts.applications.perishables.loader", "fill_missing_dates"),
    "fit_growth": ("polars_ts.applications.perishables.preprocessing", "fit_growth"),
    "remove_growth": ("polars_ts.applications.perishables.preprocessing", "remove_growth"),
    "reapply_growth": ("polars_ts.applications.perishables.preprocessing", "reapply_growth"),
    "recency_weights": ("polars_ts.applications.perishables.preprocessing", "recency_weights"),
    "select_adaptive_window": ("polars_ts.applications.perishables.preprocessing", "select_adaptive_window"),
    "apply_training_window": ("polars_ts.applications.perishables.preprocessing", "apply_training_window"),
    "classify_demand": ("polars_ts.applications.perishables.intermittent", "classify_demand"),
    "croston_forecast": ("polars_ts.applications.perishables.intermittent", "croston_forecast"),
    "sba_forecast": ("polars_ts.applications.perishables.intermittent", "sba_forecast"),
    "tsb_forecast": ("polars_ts.applications.perishables.intermittent", "tsb_forecast"),
    "intermittent_forecast": ("polars_ts.applications.perishables.intermittent", "intermittent_forecast"),
    "RestockPolicy": ("polars_ts.applications.perishables.restock", "RestockPolicy"),
    "recommend_orders": ("polars_ts.applications.perishables.restock", "recommend_orders"),
    "demand_std": ("polars_ts.applications.perishables.restock", "demand_std"),
    "cold_start_skus": ("polars_ts.applications.perishables.cold_start", "cold_start_skus"),
    "cold_start_forecast": ("polars_ts.applications.perishables.cold_start", "cold_start_forecast"),
    "dow_profile": ("polars_ts.applications.perishables.features", "dow_profile"),
    "estimate_promo_lift": ("polars_ts.applications.perishables.features", "estimate_promo_lift"),
    "perishable_calendar_features": ("polars_ts.applications.perishables.features", "perishable_calendar_features"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
