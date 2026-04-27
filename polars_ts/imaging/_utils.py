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
    groups = df.group_by(id_col, maintain_order=True).agg(pl.col(target_col))
    return {str(row[id_col]): np.array(row[target_col]) for row in groups.iter_rows(named=True)}
