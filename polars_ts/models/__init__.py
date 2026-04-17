from typing import Any

_BASELINE_NAMES = {"naive_forecast", "seasonal_naive_forecast", "moving_average_forecast", "fft_forecast"}


def __getattr__(name: str) -> Any:
    if name == "SCUM":
        try:
            from polars_ts.models.scum import SCUM
        except ImportError:
            raise ImportError(
                "statsforecast is required for SCUM. " "Install it with: pip install polars-timeseries[forecast]"
            ) from None
        return SCUM
    if name in _BASELINE_NAMES:
        from polars_ts.models import baselines

        return getattr(baselines, name)
    raise AttributeError(f"module 'polars_ts.models' has no attribute {name!r}")


__all__ = ["SCUM", "naive_forecast", "seasonal_naive_forecast", "moving_average_forecast", "fft_forecast"]
