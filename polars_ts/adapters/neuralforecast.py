"""Adapter for neuralforecast (N-BEATS, N-HiTS, PatchTST, etc.)."""

from __future__ import annotations

import polars as pl


def to_neuralforecast(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Convert a polars-ts DataFrame to neuralforecast format.

    neuralforecast expects columns named exactly ``unique_id``, ``ds``,
    and ``y``. This function renames columns as needed and ensures
    the time column is a proper datetime type.

    """
    result = (
        df.rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
        if (id_col != "unique_id" or time_col != "ds" or target_col != "y")
        else df.clone()
    )

    # Ensure ds is Datetime (neuralforecast requires it)
    if result["ds"].dtype == pl.Date:
        result = result.with_columns(pl.col("ds").cast(pl.Datetime("ms")))

    return result.sort("unique_id", "ds")


def from_neuralforecast(
    result_df: pl.DataFrame,
    _target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Convert neuralforecast output back to polars-ts format.

    neuralforecast returns DataFrames with ``unique_id``, ``ds``, and
    model-named forecast columns. This renames back and selects
    the first forecast column as ``y_hat``.

    """
    # Find forecast columns (everything except unique_id and ds)
    forecast_cols = [c for c in result_df.columns if c not in ("unique_id", "ds")]

    if not forecast_cols:
        raise ValueError("No forecast columns found in neuralforecast output")

    # Use first forecast column as y_hat
    renamed = (
        result_df.rename({"unique_id": id_col, "ds": time_col})
        if (id_col != "unique_id" or time_col != "ds")
        else result_df
    )

    return renamed.select(id_col, time_col, pl.col(forecast_cols[0]).alias("y_hat")).sort(id_col, time_col)
