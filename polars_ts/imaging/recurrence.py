"""Recurrence plot imaging and Recurrence Quantification Analysis (RQA).

Converts time series to 2D binary/grayscale recurrence plot images
and extracts RQA features (recurrence rate, determinism, laminarity,
entropy, trapping time).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from polars_ts.imaging._utils import extract_series


def _recurrence_matrix(
    x: np.ndarray,
    threshold: float | None,
    metric: str,
    normalize: bool,
) -> np.ndarray:
    """Compute recurrence plot for a single 1D series."""
    if normalize:
        std = x.std()
        if std > 0:
            x = (x - x.mean()) / std
        else:
            x = x - x.mean()

    from scipy.spatial.distance import cdist

    X = x.reshape(-1, 1)
    D: np.ndarray = cdist(X, X, metric=cast(Any, metric))

    if threshold is not None:
        return (D <= threshold).astype(np.float64)
    return D


def to_recurrence_plot(
    df: pl.DataFrame,
    threshold: float | None = 0.1,
    metric: str = "euclidean",
    normalize: bool = True,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to recurrence plot images.

    For each series, computes the pairwise distance matrix between all
    timepoints and optionally binarises it with a threshold.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    threshold
        Binarisation threshold. Points closer than this are marked as
        recurrent (1). Set to ``None`` for a grayscale distance matrix.
    metric
        Point-wise distance metric (any metric supported by
        ``scipy.spatial.distance.cdist``).
    normalize
        Z-normalize each series before computing distances.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a square 2D numpy array (n x n).

    """
    series = extract_series(df, id_col, target_col)
    return {sid: _recurrence_matrix(vals, threshold, metric, normalize) for sid, vals in series.items()}


def _diagonal_lines(R: np.ndarray) -> list[int]:
    """Extract lengths of diagonal lines (excluding main diagonal)."""
    n = R.shape[0]
    lengths: list[int] = []
    for k in range(1, n):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val > 0.5:
                length += 1
            elif length > 0:
                lengths.append(length)
                length = 0
        if length > 0:
            lengths.append(length)
    return lengths


def _vertical_lines(R: np.ndarray) -> list[int]:
    """Extract lengths of vertical lines."""
    n = R.shape[0]
    lengths: list[int] = []
    for col in range(n):
        length = 0
        for row in range(n):
            if R[row, col] > 0.5:
                length += 1
            elif length > 0:
                lengths.append(length)
                length = 0
        if length > 0:
            lengths.append(length)
    return lengths


def rqa_features(R: np.ndarray, l_min: int = 2) -> dict[str, float]:
    """Extract Recurrence Quantification Analysis features from a recurrence plot.

    Parameters
    ----------
    R
        Square binary recurrence plot (values 0 or 1).
    l_min
        Minimum line length to count for determinism and laminarity.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``recurrence_rate``, ``determinism``,
        ``laminarity``, ``mean_diagonal``, ``mean_vertical``, ``entropy``.

    """
    n = R.shape[0]
    total = n * n

    # Recurrence rate
    rr = float(R.sum()) / total if total > 0 else 0.0

    # Diagonal lines
    diag_lengths = _diagonal_lines(R)
    long_diags = [dl for dl in diag_lengths if dl >= l_min]
    det_points = sum(long_diags)
    all_recurrent = int(R.sum()) - n  # exclude main diagonal
    determinism = det_points / all_recurrent if all_recurrent > 0 else 0.0
    mean_diag = float(np.mean(long_diags)) if long_diags else 0.0

    # Vertical lines
    vert_lengths = _vertical_lines(R)
    long_verts = [vl for vl in vert_lengths if vl >= l_min]
    lam_points = sum(long_verts)
    total_vert = sum(vert_lengths)
    laminarity = lam_points / total_vert if total_vert > 0 else 0.0
    mean_vert = float(np.mean(long_verts)) if long_verts else 0.0

    # Entropy of diagonal line length distribution
    if long_diags:
        counts = np.bincount(long_diags)
        probs = counts[counts > 0] / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    return {
        "recurrence_rate": rr,
        "determinism": determinism,
        "laminarity": laminarity,
        "mean_diagonal": mean_diag,
        "mean_vertical": mean_vert,
        "entropy": entropy,
    }
