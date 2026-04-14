from pathlib import Path

import polars as pl
import polars_ts_rs as _rs_mod
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function
from polars_ts.distance import compute_pairwise_distance
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_dtw_multi,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_msm_multi,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)

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


def __getattr__(name: str):
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
    "mann_kendall",
    "sens_slope",
    "cusum",
    "fourier_decomposition",
    "seasonal_decomposition",
    "seasonal_decompose_features",
    "Metrics",
    "SCUM",
]
