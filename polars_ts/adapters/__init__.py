"""Integration adapters for DL/RL frameworks. Closes #48."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "to_neuralforecast": ("polars_ts.adapters.neuralforecast", "to_neuralforecast"),
    "from_neuralforecast": ("polars_ts.adapters.neuralforecast", "from_neuralforecast"),
    "to_pytorch_forecasting": ("polars_ts.adapters.pytorch_forecasting", "to_pytorch_forecasting"),
    "from_pytorch_forecasting": ("polars_ts.adapters.pytorch_forecasting", "from_pytorch_forecasting"),
    "to_hf_dataset": ("polars_ts.adapters.huggingface", "to_hf_dataset"),
    "ForecastEnv": ("polars_ts.adapters.rl_env", "ForecastEnv"),
    "to_chronos_embeddings": ("polars_ts.adapters.embeddings", "to_chronos_embeddings"),
    "to_moment_embeddings": ("polars_ts.adapters.embeddings", "to_moment_embeddings"),
    "foundation_forecast": ("polars_ts.adapters.foundation_forecast", "foundation_forecast"),
    "ChronosForecaster": ("polars_ts.adapters.foundation_forecast", "ChronosForecaster"),
    "TimesFMForecaster": ("polars_ts.adapters.foundation_forecast", "TimesFMForecaster"),
    "MoiraiForecaster": ("polars_ts.adapters.foundation_forecast", "MoiraiForecaster"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
