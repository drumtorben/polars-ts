from polars_ts.metrics import Metrics  # noqa

from pathlib import Path
from polars_ts_rs.polars_ts_rs import compute_pairwise_dtw, compute_pairwise_msm, compute_pairwise_ddtw, compute_pairwise_wdtw

__all__ = [
    "compute_pairwise_dtw",
    "compute_pairwise_ddtw",
    "compute_pairwise_wdtw",
    "compute_pairwise_msm",
    # plus any pure-Python symbols you want to expose
]

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parent


def mann_kendall(expr: IntoExpr) -> pl.Expr:
    """Mann-Kendall test for expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="mann_kendall",
        args=expr,
        is_elementwise=False,
    )
