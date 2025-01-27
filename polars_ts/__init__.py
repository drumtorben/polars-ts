from polars_ts.metrics import Metrics  # noqa

from pathlib import Path

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