"""Streaming / online learning support for polars-ts.

Provides incremental model updates for streaming time series data:
- StreamingETS: Online exponential smoothing (SES, Holt, Holt-Winters)
- StreamingKalmanFilter: Online Kalman filter with single-observation updates
- StreamingGlobalForecaster: Incremental global model with SGD-compatible estimators
- SlidingWindowManager: Memory-efficient windowed state management
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "StreamingETS": ("polars_ts.streaming.ets", "StreamingETS"),
    "StreamingKalmanFilter": ("polars_ts.streaming.kalman", "StreamingKalmanFilter"),
    "StreamingGlobalForecaster": ("polars_ts.streaming.global_model", "StreamingGlobalForecaster"),
    "SlidingWindowManager": ("polars_ts.streaming.window", "SlidingWindowManager"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
