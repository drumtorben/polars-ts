"""DTW Barycentric Averaging (DBA) for time series centroid computation.

Implements the iterative averaging algorithm from:
Petitjean, F. et al. (2011). *A global averaging method for dynamic time
warping*. Pattern Recognition.
"""

from __future__ import annotations

import numpy as np


def _dtw_alignment_path(s: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """Compute full DTW cost matrix and return the optimal alignment path."""
    n, m = len(s), len(t)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = (s[i - 1] - t[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtrack from (n, m) to (1, 1) in the 1-indexed cost matrix
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i >= 1 and j >= 1:
        path.append((i - 1, j - 1))  # convert to 0-indexed
        if i == 1 and j == 1:
            break
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            candidates = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            argmin = int(np.argmin(candidates))
            if argmin == 0:
                i, j = i - 1, j - 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()
    return path


def dba(
    series: list[np.ndarray],
    max_iter: int = 30,
    tol: float = 1e-5,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the DTW Barycentric Average of a set of time series.

    Parameters
    ----------
    series
        List of 1-D numpy arrays (may differ in length).
    max_iter
        Maximum number of refinement iterations.
    tol
        Convergence threshold on the mean absolute change.
    init
        Initial centroid estimate. If *None*, uses the medoid
        (series with minimum total DTW distance to all others).

    Returns
    -------
    np.ndarray
        The DBA centroid.

    """
    if len(series) == 0:
        return np.array([])
    if len(series) == 1:
        return series[0].copy()

    # Initialise centroid
    if init is not None:
        centroid = init.copy().astype(np.float64)
    else:
        # Use the medoid as initial centroid
        centroid = _medoid_init(series)

    for _ in range(max_iter):
        new_centroid = _dba_update(centroid, series)
        change = np.mean(np.abs(new_centroid - centroid))
        centroid = new_centroid
        if change < tol:
            break

    return centroid


def _medoid_init(series: list[np.ndarray]) -> np.ndarray:
    """Pick the series with minimum total squared DTW distance as init."""
    n = len(series)
    if n <= 2:
        return series[0].copy().astype(np.float64)

    # Approximate: use sum of squared Euclidean distance on truncated series
    max_len = max(len(s) for s in series)
    padded = []
    for s in series:
        if len(s) < max_len:
            padded.append(np.pad(s.astype(np.float64), (0, max_len - len(s))))
        else:
            padded.append(s.astype(np.float64))
    stacked = np.array(padded)
    # Sum of squared distances to all others
    costs = np.array([np.sum((stacked - stacked[i]) ** 2) for i in range(n)])
    return padded[int(np.argmin(costs))]


def _dba_update(centroid: np.ndarray, series: list[np.ndarray]) -> np.ndarray:
    """One DBA refinement step: align all series to centroid, average."""
    c_len = len(centroid)
    total = np.zeros(c_len)
    counts = np.zeros(c_len)

    for s in series:
        path = _dtw_alignment_path(centroid, s)
        for ci, si in path:
            total[ci] += s[si]
            counts[ci] += 1

    # Avoid division by zero
    mask = counts > 0
    result = np.zeros(c_len)
    result[mask] = total[mask] / counts[mask]
    return result
