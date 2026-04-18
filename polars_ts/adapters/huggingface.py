"""Adapter for HuggingFace time series models (Chronos, TimesFM, etc.)."""

from __future__ import annotations

from typing import Any

import polars as pl


def to_hf_dataset(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Any:
    """Convert a polars-ts DataFrame to a HuggingFace Dataset.

    Creates a ``datasets.Dataset`` with one row per time series,
    where the target values are stored as a list (the format expected
    by HuggingFace time series models like Chronos).

    Requires the ``datasets`` package.

    Returns
    -------
    datasets.Dataset
        HuggingFace Dataset with columns ``["id", "target", "start"]``.

    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Install it with: pip install datasets") from None

    sorted_df = df.sort(id_col, time_col)

    records: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        values = group_df[target_col].to_list()
        start = str(group_df[time_col][0])
        records.append({"id": str(gid), "target": values, "start": start})

    return Dataset.from_list(records)
