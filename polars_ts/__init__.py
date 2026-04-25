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
    if name in {"hdbscan_cluster", "dbscan_cluster"}:
        from polars_ts.clustering import density as _density

        return getattr(_density, name)
    if name in {"clara", "clarans"}:
        from polars_ts.clustering import scalable as _scalable

        return getattr(_scalable, name)
    if name == "kmeans_dba":
        from polars_ts.clustering.kmeans import kmeans_dba

        return kmeans_dba
    if name == "TimeSeriesKMeans":
        from polars_ts.clustering.kmeans import TimeSeriesKMeans

        return TimeSeriesKMeans
    if name == "agglomerative_cluster":
        from polars_ts.clustering.hierarchical import agglomerative_cluster

        return agglomerative_cluster
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
        "ses_forecast",
        "holt_forecast",
        "holt_winters_forecast",
    }:
        from polars_ts import models as _models

        return getattr(_models, name)
    if name in {"pelt", "bocpd", "regime_detect"}:
        from polars_ts import changepoint as _cp

        return getattr(_cp, name)
    if name in {"garch_fit", "garch_forecast", "GARCHResult"}:
        from polars_ts import volatility as _vol

        return getattr(_vol, name)
    if name in {"var_fit", "var_forecast", "granger_causality", "VARResult"}:
        from polars_ts import var_model as _var

        return getattr(_var, name)
    if name == "reconcile":
        from polars_ts.reconciliation import reconcile

        return reconcile
    if name in {
        "to_neuralforecast",
        "from_neuralforecast",
        "to_pytorch_forecasting",
        "from_pytorch_forecasting",
        "to_hf_dataset",
        "ForecastEnv",
    }:
        from polars_ts import adapters as _adapt

        return getattr(_adapt, name)
    if name in {"target_encode", "holiday_features", "interaction_features", "time_embeddings"}:
        from polars_ts.features import advanced as _adv

        return getattr(_adv, name)
    if name in {"bias_detect", "bias_correct"}:
        from polars_ts import bias as _bias

        return getattr(_bias, name)
    if name in {"calibration_table", "pit_histogram", "reliability_diagram"}:
        from polars_ts import calibration as _cal

        return getattr(_cal, name)
    if name == "permutation_importance":
        from polars_ts.importance import permutation_importance

        return permutation_importance
    if name == "isolation_forest_detect":
        from polars_ts.anomaly_forest import isolation_forest_detect

        return isolation_forest_detect
    if name == "impute":
        from polars_ts.imputation import impute

        return impute
    if name in {"detect_outliers", "treat_outliers"}:
        from polars_ts import outliers as _outliers

        return getattr(_outliers, name)
    if name == "resample":
        from polars_ts.resampling import resample

        return resample
    if name in {"acf", "pacf", "ljung_box"}:
        from polars_ts import diagnostics as _diag

        return getattr(_diag, name)
    if name == "ForecastPipeline":
        from polars_ts.pipeline import ForecastPipeline

        return ForecastPipeline
    if name == "GlobalForecaster":
        from polars_ts.global_model import GlobalForecaster

        return GlobalForecaster
    if name in {"WeightedEnsemble", "StackingForecaster"}:
        from polars_ts import ensemble as _ens

        return getattr(_ens, name)
    if name in {"QuantileRegressor", "conformal_interval", "EnbPI"}:
        from polars_ts import probabilistic as _prob

        return getattr(_prob, name)
    if name in {"arima_fit", "arima_forecast", "auto_arima"}:
        from polars_ts import models as _models

        return getattr(_models, name)
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
    "pelt",
    "bocpd",
    "regime_detect",
    "garch_fit",
    "garch_forecast",
    "GARCHResult",
    "var_fit",
    "var_forecast",
    "granger_causality",
    "VARResult",
    "reconcile",
    "to_neuralforecast",
    "from_neuralforecast",
    "to_pytorch_forecasting",
    "from_pytorch_forecasting",
    "to_hf_dataset",
    "ForecastEnv",
    "target_encode",
    "holiday_features",
    "interaction_features",
    "time_embeddings",
    "bias_detect",
    "bias_correct",
    "calibration_table",
    "pit_histogram",
    "reliability_diagram",
    "permutation_importance",
    "isolation_forest_detect",
    "ses_forecast",
    "holt_forecast",
    "holt_winters_forecast",
    "impute",
    "detect_outliers",
    "treat_outliers",
    "resample",
    "acf",
    "pacf",
    "ljung_box",
    "ForecastPipeline",
    "GlobalForecaster",
    "WeightedEnsemble",
    "StackingForecaster",
    "QuantileRegressor",
    "conformal_interval",
    "EnbPI",
    "arima_fit",
    "arima_forecast",
    "auto_arima",
    "hdbscan_cluster",
    "dbscan_cluster",
    "clara",
    "clarans",
    "kmeans_dba",
    "TimeSeriesKMeans",
    "agglomerative_cluster",
]
