"""Probabilistic forecasting: quantile regression and conformal prediction."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "QuantileRegressor": ("polars_ts.probabilistic.quantile_regression", "QuantileRegressor"),
    "conformal_interval": ("polars_ts.probabilistic.conformal", "conformal_interval"),
    "EnbPI": ("polars_ts.probabilistic.conformal", "EnbPI"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
