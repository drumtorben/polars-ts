"""Forecast ensembling: weighted combination and stacking."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "WeightedEnsemble": ("polars_ts.ensemble.weighted", "WeightedEnsemble"),
    "StackingForecaster": ("polars_ts.ensemble.stacking", "StackingForecaster"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
