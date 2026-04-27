"""Markov Transition Field (MTF) imaging for time series.

Discretises values into quantile bins, estimates a Markov transition
matrix, and maps transition probabilities back onto the time axis.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_ts.imaging._utils import extract_series


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a quantile bin (0 to n_bins-1)."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    # np.digitize returns 1-based; clip to [0, n_bins-1]
    bins = np.digitize(x, edges[1:-1], right=True)
    return np.clip(bins, 0, n_bins - 1)


def _mtf_matrix(x: np.ndarray, n_bins: int, image_size: int | None) -> np.ndarray:
    """Compute MTF for a single 1D series."""
    bins = _quantile_bins(x, n_bins)

    # Transition matrix
    T = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(len(bins) - 1):
        T[bins[i], bins[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T /= row_sums

    # Build MTF: M[i,j] = T[bin(x_i), bin(x_j)]
    n = len(bins)
    M = T[bins[:, None], bins[None, :]]

    # Optional PAA downsampling
    if image_size is not None and image_size < n:
        indices = np.array_split(np.arange(n), image_size)
        M_down = np.zeros((image_size, image_size))
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                M_down[i, j] = M[np.ix_(idx_i, idx_j)].mean()
        return M_down

    return M


def to_mtf(
    df: pl.DataFrame,
    n_bins: int = 8,
    image_size: int | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to Markov Transition Field images.

    Quantises values into ``n_bins`` bins, computes the Markov transition
    matrix, then builds an n x n image where pixel (i, j) is the
    transition probability from bin(x_i) to bin(x_j).

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    n_bins
        Number of quantile bins for discretisation.
    image_size
        Output image dimension. ``None`` for full resolution (n x n).
        Smaller values use Piecewise Aggregate Approximation (PAA).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a square 2D array with values in [0, 1].

    """
    series = extract_series(df, id_col, target_col)
    return {sid: _mtf_matrix(vals, n_bins, image_size) for sid, vals in series.items()}
