from typing import Any

_BASELINE_NAMES = {"naive_forecast", "seasonal_naive_forecast", "moving_average_forecast", "fft_forecast"}
_MULTISTEP_NAMES = {"RecursiveForecaster", "DirectForecaster"}
_ES_NAMES = {"ses_forecast", "holt_forecast", "holt_winters_forecast"}

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
    if name in _MULTISTEP_NAMES:
        from polars_ts.models import multistep

        return getattr(multistep, name)
    if name in _ES_NAMES:
        from polars_ts.models import exponential_smoothing

        return getattr(exponential_smoothing, name)
    raise AttributeError(f"module 'polars_ts.models' has no attribute {name!r}")


__all__ = [
    "SCUM",
    "naive_forecast",
    "seasonal_naive_forecast",
    "moving_average_forecast",
    "fft_forecast",
    "RecursiveForecaster",
    "DirectForecaster",
    "ses_forecast",
    "holt_forecast",
    "holt_winters_forecast",
]
