"""Shared utilities for deep learning classifiers."""

from __future__ import annotations

import numpy as np
import polars as pl


def extract_classification_data(
    df: pl.DataFrame,
    id_col: str,
    target_col: str,
    time_col: str,
    label_col: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Extract padded series and optional labels.

    Returns
    -------
    ids
        Series identifiers.
    X
        Padded array of shape ``(n_series, max_len)``.
    y
        Integer labels of shape ``(n_series,)`` or ``None`` if no label_col.

    """
    sort_cols = [id_col, time_col] if time_col in df.columns else [id_col]
    sorted_df = df.sort(sort_cols)

    ids: list[str] = []
    arrays: list[np.ndarray] = []
    labels: list[str] = []

    for key, group in sorted_df.group_by(id_col, maintain_order=True):
        sid = key[0] if isinstance(key, tuple) else key
        ids.append(str(sid))
        arrays.append(group[target_col].to_numpy().astype(np.float64))
        if label_col is not None and label_col in group.columns:
            labels.append(str(group[label_col][0]))

    max_len = max(a.shape[0] for a in arrays)
    X = np.zeros((len(arrays), max_len), dtype=np.float64)
    for i, a in enumerate(arrays):
        X[i, : a.shape[0]] = a

    y_arr = None
    if labels:
        unique_labels = sorted(set(labels))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_arr = np.array([label_map[lbl] for lbl in labels])

    return ids, X, y_arr
