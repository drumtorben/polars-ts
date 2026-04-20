"""Forecast ensembling: weighted combination and stacking.

Implements ensemble strategies from Ch 9 of
"Modern Time Series Forecasting with Python" (2nd Ed.).
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "WeightedEnsemble":
        from polars_ts.ensemble.weighted import WeightedEnsemble

        return WeightedEnsemble
    if name == "StackingForecaster":
        from polars_ts.ensemble.stacking import StackingForecaster

        return StackingForecaster
    raise AttributeError(f"module 'polars_ts.ensemble' has no attribute {name!r}")


__all__ = ["WeightedEnsemble", "StackingForecaster"]
