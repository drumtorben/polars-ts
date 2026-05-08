"""Model registry and experiment tracking."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "ModelRegistry": ("polars_ts.registry.registry", "ModelRegistry"),
    "Experiment": ("polars_ts.registry.experiment", "Experiment"),
    "Run": ("polars_ts.registry.experiment", "Run"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
