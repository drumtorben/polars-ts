from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"KalmanFilter", "kalman_filter"}:
        from polars_ts.bayesian.kalman import KalmanFilter, kalman_filter

        if name == "KalmanFilter":
            return KalmanFilter
        return kalman_filter
    if name == "UnscentedKalmanFilter":
        from polars_ts.bayesian.ukf import UnscentedKalmanFilter

        return UnscentedKalmanFilter
    if name == "EnsembleKalmanFilter":
        from polars_ts.bayesian.enkf import EnsembleKalmanFilter

        return EnsembleKalmanFilter
    raise AttributeError(f"module 'polars_ts.bayesian' has no attribute {name!r}")


__all__ = [
    "KalmanFilter",
    "kalman_filter",
    "UnscentedKalmanFilter",
    "EnsembleKalmanFilter",
]
