from polars_ts.metrics import Metrics  # noqa

from pathlib import Path
from polars_ts_rs.polars_ts_rs import compute_pairwise_dtw

__all__ = [
    "compute_pairwise_dtw",
    # plus any pure-Python symbols you want to expose
]

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent


def mann_kendall(expr: IntoExpr) -> pl.Expr:
    """Mann-Kendall test for expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="mann_kendall",
        args=expr,
        is_elementwise=False,
    )