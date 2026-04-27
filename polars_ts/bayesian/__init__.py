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
    if name in {"BSTS", "bsts_fit", "bsts_forecast"}:
        from polars_ts.bayesian.bsts import BSTS, bsts_fit, bsts_forecast

        if name == "BSTS":
            return BSTS
        if name == "bsts_fit":
            return bsts_fit
        return bsts_forecast
    raise AttributeError(f"module 'polars_ts.bayesian' has no attribute {name!r}")


__all__ = [
    "KalmanFilter",
    "kalman_filter",
    "UnscentedKalmanFilter",
    "EnsembleKalmanFilter",
    "BSTS",
    "bsts_fit",
    "bsts_forecast",
]
