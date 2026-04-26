from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "fourier_decomposition": ("polars_ts.decomposition.fourier_decomposition", "fourier_decomposition"),
    "seasonal_decomposition": ("polars_ts.decomposition.seasonal_decomposition", "seasonal_decomposition"),
    "seasonal_decompose_features": (
        "polars_ts.decomposition.seasonal_decompose_features",
        "seasonal_decompose_features",
    ),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
