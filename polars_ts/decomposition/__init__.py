from typing import Any


def __getattr__(name: str) -> Any:
    if name == "fourier_decomposition":
        from polars_ts.decomposition.fourier_decomposition import fourier_decomposition

        return fourier_decomposition
    if name == "seasonal_decomposition":
        from polars_ts.decomposition.seasonal_decomposition import seasonal_decomposition

        return seasonal_decomposition
    if name == "seasonal_decompose_features":
        from polars_ts.decomposition.seasonal_decompose_features import seasonal_decompose_features

        return seasonal_decompose_features
    raise AttributeError(f"module 'polars_ts.decomposition' has no attribute {name!r}")


__all__ = ["fourier_decomposition", "seasonal_decomposition", "seasonal_decompose_features"]
