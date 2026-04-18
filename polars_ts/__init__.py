from pathlib import Path
from typing import Any

import polars as pl
import polars_ts_rs as _rs_mod
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_dtw_multi,
    compute_pairwise_edr,
    compute_pairwise_erp,
    compute_pairwise_frechet,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_msm_multi,
    compute_pairwise_sbd,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)

from polars_ts.distance import compute_pairwise_distance

PLUGIN_PATH = Path(_rs_mod.__file__).parent


def mann_kendall(expr: IntoExpr) -> pl.Expr:
    """Mann-Kendall test for expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="mann_kendall",
        args=expr,
        is_elementwise=False,
    )


def sens_slope(expr: IntoExpr) -> pl.Expr:
    """Sen's slope estimator (median of pairwise slopes)."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="sens_slope",
        args=expr,
        is_elementwise=False,
    )


def __getattr__(name: str) -> Any:
    if name == "Metrics":
        from polars_ts.metrics import Metrics

        return Metrics
    if name == "SCUM":
        from polars_ts.models import SCUM

        return SCUM
    if name == "fourier_decomposition":
        from polars_ts.decomposition.fourier_decomposition import fourier_decomposition

        return fourier_decomposition
    if name == "seasonal_decomposition":
        from polars_ts.decomposition.seasonal_decomposition import seasonal_decomposition

        return seasonal_decomposition
    if name == "seasonal_decompose_features":
        from polars_ts.decomposition.seasonal_decompose_features import seasonal_decompose_features

        return seasonal_decompose_features
    if name == "cusum":
        from polars_ts.changepoint.cusum import cusum

        return cusum
    if name == "kmedoids":
        from polars_ts.clustering.kmedoids import kmedoids

        return kmedoids
    if name == "knn_classify":
        from polars_ts.classification.knn import knn_classify

        return knn_classify
    if name == "TimeSeriesKNNClassifier":
        from polars_ts.classification.knn import TimeSeriesKNNClassifier

        return TimeSeriesKNNClassifier
    if name == "KShapeClassifier":
        from polars_ts.classification.kshape_classifier import KShapeClassifier

        return KShapeClassifier
    if name == "TimeSeriesKMedoids":
        from polars_ts.clustering.kmedoids import TimeSeriesKMedoids

        return TimeSeriesKMedoids
    if name == "KShape":
        from polars_ts.clustering.kshape import KShape

        return KShape
    if name == "silhouette_score":
        from polars_ts.clustering.evaluation import silhouette_score

        return silhouette_score
    if name == "silhouette_samples":
        from polars_ts.clustering.evaluation import silhouette_samples

        return silhouette_samples
    if name == "davies_bouldin_score":
        from polars_ts.clustering.evaluation import davies_bouldin_score

        return davies_bouldin_score
    if name == "calinski_harabasz_score":
        from polars_ts.clustering.evaluation import calinski_harabasz_score

        return calinski_harabasz_score
    if name in {"lag_features", "rolling_features", "calendar_features", "fourier_features"}:
        from polars_ts import features as _feat

        return getattr(_feat, name)
    if name in {
        "log_transform",
        "inverse_log_transform",
        "boxcox_transform",
        "inverse_boxcox_transform",
        "difference",
        "undifference",
    }:
        from polars_ts import transforms as _tr

        return getattr(_tr, name)
    if name in {"expanding_window_cv", "sliding_window_cv", "rolling_origin_cv"}:
        from polars_ts import validation as _val

        return getattr(_val, name)
    if name in {"mae", "rmse", "mape", "smape", "mase", "crps"}:
        from polars_ts.metrics import forecast as _fm

        return getattr(_fm, name)
    if name in {
        "naive_forecast",
        "seasonal_naive_forecast",
        "moving_average_forecast",
        "fft_forecast",
        "RecursiveForecaster",
        "DirectForecaster",
    }:
        from polars_ts import models as _models

        return getattr(_models, name)
    if name == "ForecastPipeline":
        from polars_ts.pipeline import ForecastPipeline

        return ForecastPipeline
    if name == "GlobalForecaster":
        from polars_ts.global_model import GlobalForecaster

        return GlobalForecaster
    raise AttributeError(f"module 'polars_ts' has no attribute {name!r}")


__all__ = [
    "compute_pairwise_distance",
    "compute_pairwise_dtw",
    "compute_pairwise_ddtw",
    "compute_pairwise_wdtw",
    "compute_pairwise_msm",
    "compute_pairwise_dtw_multi",
    "compute_pairwise_msm_multi",
    "compute_pairwise_erp",
    "compute_pairwise_lcss",
    "compute_pairwise_twe",
    "compute_pairwise_sbd",
    "compute_pairwise_frechet",
    "compute_pairwise_edr",
    "mann_kendall",
    "sens_slope",
    "cusum",
    "fourier_decomposition",
    "seasonal_decomposition",
    "seasonal_decompose_features",
    "Metrics",
    "SCUM",
    "kmedoids",
    "knn_classify",
    "TimeSeriesKNNClassifier",
    "KShapeClassifier",
    "TimeSeriesKMedoids",
    "KShape",
    "silhouette_score",
    "silhouette_samples",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "lag_features",
    "rolling_features",
    "calendar_features",
    "fourier_features",
    "log_transform",
    "inverse_log_transform",
    "boxcox_transform",
    "inverse_boxcox_transform",
    "difference",
    "undifference",
    "expanding_window_cv",
    "sliding_window_cv",
    "rolling_origin_cv",
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "crps",
    "naive_forecast",
    "seasonal_naive_forecast",
    "moving_average_forecast",
    "fft_forecast",
    "RecursiveForecaster",
    "DirectForecaster",
    "ForecastPipeline",
    "GlobalForecaster",
]
