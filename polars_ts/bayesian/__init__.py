from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "KalmanFilter": ("polars_ts.bayesian.kalman", "KalmanFilter"),
    "kalman_filter": ("polars_ts.bayesian.kalman", "kalman_filter"),
    "UnscentedKalmanFilter": ("polars_ts.bayesian.ukf", "UnscentedKalmanFilter"),
    "EnsembleKalmanFilter": ("polars_ts.bayesian.enkf", "EnsembleKalmanFilter"),
    "BSTS": ("polars_ts.bayesian.bsts", "BSTS"),
    "bsts_fit": ("polars_ts.bayesian.bsts", "bsts_fit"),
    "bsts_forecast": ("polars_ts.bayesian.bsts", "bsts_forecast"),
    "GaussianProcessTS": ("polars_ts.bayesian.gp", "GaussianProcessTS"),
    "gp_forecast": ("polars_ts.bayesian.gp", "gp_forecast"),
    "BayesianAnomalyDetector": ("polars_ts.bayesian.anomaly", "BayesianAnomalyDetector"),
    "bayesian_anomaly_score": ("polars_ts.bayesian.anomaly", "bayesian_anomaly_score"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
