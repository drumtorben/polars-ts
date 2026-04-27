"""Target transform subpackage for time series data."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "log_transform": ("polars_ts.transforms.log", "log_transform"),
    "inverse_log_transform": ("polars_ts.transforms.log", "inverse_log_transform"),
    "boxcox_transform": ("polars_ts.transforms.boxcox", "boxcox_transform"),
    "inverse_boxcox_transform": ("polars_ts.transforms.boxcox", "inverse_boxcox_transform"),
    "difference": ("polars_ts.transforms.differencing", "difference"),
    "undifference": ("polars_ts.transforms.differencing", "undifference"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
