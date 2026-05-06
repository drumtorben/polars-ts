from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "CausalImpact": ("polars_ts.causal.causal_impact", "CausalImpact"),
    "causal_impact": ("polars_ts.causal.causal_impact", "causal_impact"),
    "CausalImpactResult": ("polars_ts.causal.causal_impact", "CausalImpactResult"),
    "SyntheticControl": ("polars_ts.causal.synthetic_control", "SyntheticControl"),
    "synthetic_control": ("polars_ts.causal.synthetic_control", "synthetic_control"),
    "SyntheticControlResult": ("polars_ts.causal.synthetic_control", "SyntheticControlResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
