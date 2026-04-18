"""Adapter for pytorch-forecasting (TFT, DeepAR, etc.)."""

from __future__ import annotations

from typing import Any

import polars as pl


def to_pytorch_forecasting(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    time_idx_col: str = "time_idx",
) -> dict[str, Any]:
    """Convert a polars-ts DataFrame to pytorch-forecasting format.

    Returns a dict with ``"data"`` (a pandas DataFrame) and
    ``"metadata"`` needed to construct a ``TimeSeriesDataSet``.

    pytorch-forecasting requires a pandas DataFrame with a numeric
    time index column.

    """
    sorted_df = df.sort(id_col, time_col)

    # Add numeric time index per group
    result = sorted_df.with_columns(
        pl.col(time_col).rank("ordinal").over(id_col).cast(pl.Int64).sub(1).alias(time_idx_col)
    )

    pandas_df = result.to_pandas()

    return {
        "data": pandas_df,
        "metadata": {
            "time_idx": time_idx_col,
            "target": target_col,
            "group_ids": [id_col],
        },
    }


def from_pytorch_forecasting(
    predictions: Any,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Convert pytorch-forecasting predictions back to polars-ts format.

    Parameters
    ----------
    predictions
        A pandas DataFrame or numpy array of predictions. If a
        DataFrame, it should have id/time columns plus prediction
        values. If a numpy array, a simple DataFrame is returned.

    """
    import numpy as np

    if isinstance(predictions, np.ndarray):
        return pl.DataFrame({"y_hat": predictions.flatten().tolist()})

    # Assume pandas DataFrame
    pdf = predictions
    result = pl.from_pandas(pdf)

    # Rename prediction column to y_hat if needed
    pred_cols = [c for c in result.columns if c not in (id_col, time_col, "time_idx")]
    if pred_cols and "y_hat" not in result.columns:
        result = result.rename({pred_cols[0]: "y_hat"})

    select_cols = [c for c in [id_col, time_col, "y_hat"] if c in result.columns]
    return result.select(select_cols)
