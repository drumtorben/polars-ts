from typing import Any

_BASELINE_NAMES = {"naive_forecast", "seasonal_naive_forecast", "moving_average_forecast", "fft_forecast"}
_MULTISTEP_NAMES = {"RecursiveForecaster", "DirectForecaster"}
_ES_NAMES = {"ses_forecast", "holt_forecast", "holt_winters_forecast"}
_ARIMA_NAMES = {"arima_fit", "arima_forecast", "auto_arima"}
_BAYESIAN_ETS_NAMES = {"bayesian_ets", "BayesianETS", "ETSPriors"}


def __getattr__(name: str) -> Any:
    if name in _BAYESIAN_ETS_NAMES:
        from polars_ts.models import bayesian_ets as _bets

        return getattr(_bets, name)
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
    if name in _ARIMA_NAMES:
        from polars_ts.models import arima

        return getattr(arima, name)
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
    "arima_fit",
    "arima_forecast",
    "auto_arima",
    "bayesian_ets",
    "BayesianETS",
    "ETSPriors",
]
