"""Feature engineering subpackage for time series data."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "lag_features": ("polars_ts.features.lags", "lag_features"),
    "rolling_features": ("polars_ts.features.rolling", "rolling_features"),
    "calendar_features": ("polars_ts.features.calendar", "calendar_features"),
    "fourier_features": ("polars_ts.features.fourier", "fourier_features"),
    "rocket_features": ("polars_ts.features.rocket", "rocket_features"),
    "minirocket_features": ("polars_ts.features.rocket", "minirocket_features"),
}

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "lag_features":
        from polars_ts.features.lags import lag_features

        return lag_features
    if name == "covariate_lag_features":
        from polars_ts.features.lags import covariate_lag_features

        return covariate_lag_features
    if name == "rolling_features":
        from polars_ts.features.rolling import rolling_features

        return rolling_features
    if name == "calendar_features":
        from polars_ts.features.calendar import calendar_features

        return calendar_features
    if name == "fourier_features":
        from polars_ts.features.fourier import fourier_features

        return fourier_features
    if name == "rocket_features":
        from polars_ts.features.rocket import rocket_features

        return rocket_features
    if name == "minirocket_features":
        from polars_ts.features.rocket import minirocket_features

        return minirocket_features
    raise AttributeError(f"module 'polars_ts.features' has no attribute {name!r}")


__all__ = [
    "lag_features",
    "covariate_lag_features",
    "rolling_features",
    "calendar_features",
    "fourier_features",
    "rocket_features",
    "minirocket_features",
]
