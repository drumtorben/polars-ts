"""Integration adapters for DL/RL frameworks. Closes #48."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"to_neuralforecast", "from_neuralforecast"}:
        from polars_ts.adapters import neuralforecast

        return getattr(neuralforecast, name)
    if name in {"to_pytorch_forecasting", "from_pytorch_forecasting"}:
        from polars_ts.adapters import pytorch_forecasting

        return getattr(pytorch_forecasting, name)
    if name == "to_hf_dataset":
        from polars_ts.adapters.huggingface import to_hf_dataset

        return to_hf_dataset
    if name == "ForecastEnv":
        from polars_ts.adapters.rl_env import ForecastEnv

        return ForecastEnv
    if name in {"to_chronos_embeddings", "to_moment_embeddings"}:
        from polars_ts.adapters import embeddings as _emb

        return getattr(_emb, name)
    raise AttributeError(f"module 'polars_ts.adapters' has no attribute {name!r}")


__all__ = [
    "to_neuralforecast",
    "from_neuralforecast",
    "to_pytorch_forecasting",
    "from_pytorch_forecasting",
    "to_hf_dataset",
    "ForecastEnv",
    "to_chronos_embeddings",
    "to_moment_embeddings",
]
