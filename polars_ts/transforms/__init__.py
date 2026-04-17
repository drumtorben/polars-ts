"""Target transform subpackage for time series data.

Provides invertible transforms (log, Box-Cox, differencing) for
preparing target columns before modelling and restoring predictions
to the original scale.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "log_transform":
        from polars_ts.transforms.log import log_transform

        return log_transform
    if name == "inverse_log_transform":
        from polars_ts.transforms.log import inverse_log_transform

        return inverse_log_transform
    if name == "boxcox_transform":
        from polars_ts.transforms.boxcox import boxcox_transform

        return boxcox_transform
    if name == "inverse_boxcox_transform":
        from polars_ts.transforms.boxcox import inverse_boxcox_transform

        return inverse_boxcox_transform
    if name == "difference":
        from polars_ts.transforms.differencing import difference

        return difference
    if name == "undifference":
        from polars_ts.transforms.differencing import undifference

        return undifference
    raise AttributeError(f"module 'polars_ts.transforms' has no attribute {name!r}")


__all__ = [
    "log_transform",
    "inverse_log_transform",
    "boxcox_transform",
    "inverse_boxcox_transform",
    "difference",
    "undifference",
]
