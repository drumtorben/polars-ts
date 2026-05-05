"""Unified backtesting framework for time series models."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "backtest": ("polars_ts.backtesting.backtest", "backtest"),
    "compare_models": ("polars_ts.backtesting.backtest", "compare_models"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
