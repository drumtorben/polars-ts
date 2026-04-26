"""Shared utilities for the imaging module."""

from __future__ import annotations

import numpy as np
import polars as pl


def extract_series(
    df: pl.DataFrame,
    id_col: str,
    target_col: str,
) -> dict[str, np.ndarray]:
    """Group DataFrame by id_col and return dict of numpy arrays."""
    result: dict[str, np.ndarray] = {}
    for sid in df[id_col].unique(maintain_order=True).to_list():
        vals = df.filter(pl.col(id_col) == sid)[target_col].to_numpy()
        result[str(sid)] = vals
    return result
