"""Probabilistic forecasting: quantile regression and conformal prediction.

Implements prediction interval methods from Ch 16 of
"Modern Time Series Forecasting with Python" (2nd Ed.).
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "QuantileRegressor":
        from polars_ts.probabilistic.quantile_regression import QuantileRegressor

        return QuantileRegressor
    if name == "conformal_interval":
        from polars_ts.probabilistic.conformal import conformal_interval

        return conformal_interval
    if name == "EnbPI":
        from polars_ts.probabilistic.conformal import EnbPI

        return EnbPI
    raise AttributeError(f"module 'polars_ts.probabilistic' has no attribute {name!r}")


__all__ = ["QuantileRegressor", "conformal_interval", "EnbPI"]
