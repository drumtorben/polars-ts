"""Validation strategies for time series backtesting.

Provides group-aware temporal cross-validation splitters:
expanding window, sliding window, and rolling-origin CV.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "expanding_window_cv":
        from polars_ts.validation.splits import expanding_window_cv

        return expanding_window_cv
    if name == "sliding_window_cv":
        from polars_ts.validation.splits import sliding_window_cv

        return sliding_window_cv
    if name == "rolling_origin_cv":
        from polars_ts.validation.splits import rolling_origin_cv

        return rolling_origin_cv
    raise AttributeError(f"module 'polars_ts.validation' has no attribute {name!r}")


__all__ = [
    "expanding_window_cv",
    "sliding_window_cv",
    "rolling_origin_cv",
]
