from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "naive_forecast": ("polars_ts.models.baselines", "naive_forecast"),
    "seasonal_naive_forecast": ("polars_ts.models.baselines", "seasonal_naive_forecast"),
    "moving_average_forecast": ("polars_ts.models.baselines", "moving_average_forecast"),
    "fft_forecast": ("polars_ts.models.baselines", "fft_forecast"),
    "RecursiveForecaster": ("polars_ts.models.multistep", "RecursiveForecaster"),
    "DirectForecaster": ("polars_ts.models.multistep", "DirectForecaster"),
    "ses_forecast": ("polars_ts.models.exponential_smoothing", "ses_forecast"),
    "holt_forecast": ("polars_ts.models.exponential_smoothing", "holt_forecast"),
    "holt_winters_forecast": ("polars_ts.models.exponential_smoothing", "holt_winters_forecast"),
    "arima_fit": ("polars_ts.models.arima", "arima_fit"),
    "arima_forecast": ("polars_ts.models.arima", "arima_forecast"),
    "auto_arima": ("polars_ts.models.arima", "auto_arima"),
    "bayesian_ets": ("polars_ts.models.bayesian_ets", "bayesian_ets"),
    "BayesianETS": ("polars_ts.models.bayesian_ets", "BayesianETS"),
    "ETSPriors": ("polars_ts.models.bayesian_ets", "ETSPriors"),
}


def _load_scum():
    try:
        from polars_ts.models.scum import SCUM
    except ImportError:
        raise ImportError(
            "statsforecast is required for SCUM. Install it with: pip install polars-timeseries[forecast]"
        ) from None
    return SCUM


__getattr__, __all__ = make_getattr(_IMPORTS, __name__, overrides={"SCUM": _load_scum})
