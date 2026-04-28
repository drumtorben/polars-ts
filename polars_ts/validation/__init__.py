"""Validation strategies for time series backtesting."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "expanding_window_cv": ("polars_ts.validation.splits", "expanding_window_cv"),
    "sliding_window_cv": ("polars_ts.validation.splits", "sliding_window_cv"),
    "rolling_origin_cv": ("polars_ts.validation.splits", "rolling_origin_cv"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
