"""Streaming / online learning support for polars-ts.

Provides incremental model updates for streaming time series data:
- StreamingETS: Online exponential smoothing (SES, Holt, Holt-Winters)
- StreamingKalmanFilter: Online Kalman filter with single-observation updates
- StreamingGlobalForecaster: Incremental global model with SGD-compatible estimators
- SlidingWindowManager: Memory-efficient windowed state management
"""

from polars_ts.streaming.ets import StreamingETS
from polars_ts.streaming.global_model import StreamingGlobalForecaster
from polars_ts.streaming.kalman import StreamingKalmanFilter
from polars_ts.streaming.window import SlidingWindowManager

__all__ = [
    "StreamingETS",
    "StreamingKalmanFilter",
    "StreamingGlobalForecaster",
    "SlidingWindowManager",
]
