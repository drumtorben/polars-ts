"""Feature engineering subpackage for time series data."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "lag_features": ("polars_ts.features.lags", "lag_features"),
    "covariate_lag_features": ("polars_ts.features.lags", "covariate_lag_features"),
    "rolling_features": ("polars_ts.features.rolling", "rolling_features"),
    "calendar_features": ("polars_ts.features.calendar", "calendar_features"),
    "fourier_features": ("polars_ts.features.fourier", "fourier_features"),
    "rocket_features": ("polars_ts.features.rocket", "rocket_features"),
    "minirocket_features": ("polars_ts.features.rocket", "minirocket_features"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
