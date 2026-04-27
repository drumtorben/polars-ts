from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"KalmanFilter", "kalman_filter"}:
        from polars_ts.bayesian.kalman import KalmanFilter, kalman_filter

        if name == "KalmanFilter":
            return KalmanFilter
        return kalman_filter
    raise AttributeError(f"module 'polars_ts.bayesian' has no attribute {name!r}")


__all__ = [
    "KalmanFilter",
    "kalman_filter",
]
