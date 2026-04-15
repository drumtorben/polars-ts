"""k-Shape clustering for time series using Shape-Based Distance (SBD)."""

from __future__ import annotations

import numpy as np
import polars as pl


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-normalize a 1-D array. Returns zeros if std is zero."""
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _sbd(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Shape-Based Distance between two z-normalized series.

    Returns (distance, y_shifted) where y_shifted is y aligned to x.
    """
    ncc = np.correlate(x, y, mode="full")
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        return 0.0, y
    ncc = ncc / norm
    shift = np.argmax(ncc) - (len(y) - 1)
    y_shifted = np.roll(y, shift)
    return float(1.0 - ncc.max()), y_shifted


def _shape_extraction(cluster_series: list[np.ndarray], length: int) -> np.ndarray:
    """Extract centroid shape via eigenvalue decomposition (Paparrizos & Gravano).

    Computes the first eigenvector of the cross-correlation matrix S^T * S,
    which maximizes the sum of squared normalized cross-correlations.
    """
    if not cluster_series:
        return np.zeros(length)

    # Z-normalize all series
    normed = [_zscore(s) for s in cluster_series]

    # Build the matrix M = S^T @ S where S stacks the series
    s_matrix = np.column_stack(normed) if len(normed) > 1 else normed[0].reshape(-1, 1)
    m = s_matrix @ s_matrix.T

    # Dominant eigenvector via power iteration
    vec = np.random.default_rng(42).standard_normal(length)
    for _ in range(100):
        new_vec = m @ vec
        norm = np.linalg.norm(new_vec)
        if norm == 0:
            return np.zeros(length)
        vec = new_vec / norm

    # Pick sign that maximizes correlation with first series
    if normed and np.dot(vec, normed[0]) < 0:
        vec = -vec

    return _zscore(vec)


class KShape:
    """k-Shape clustering for time series.

    Uses Shape-Based Distance (SBD) and shape extraction to iteratively
    refine cluster centroids and assignments.

    Args:
        n_clusters: Number of clusters. Default 2.
        max_iter: Maximum number of iterations. Default 100.

    Examples:
        >>> ks = KShape(n_clusters=3)
        >>> ks.fit(df)
        >>> ks.labels_

    """

    def __init__(self, n_clusters: int = 2, max_iter: int = 100) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_: pl.DataFrame | None = None
        self.centroids_: list[np.ndarray] = []

    def fit(self, df: pl.DataFrame) -> KShape:
        """Fit k-Shape clustering.

        Args:
            df: DataFrame with columns ``unique_id`` and ``y``.

        Returns:
            self, with ``labels_`` and ``centroids_`` populated.

        """
        # Extract series as numpy arrays
        ids: list[str] = []
        series_list: list[np.ndarray] = []
        for uid in sorted(df["unique_id"].unique().cast(pl.String).to_list()):
            vals = df.filter(pl.col("unique_id").cast(pl.String) == uid)["y"].to_numpy()
            ids.append(uid)
            series_list.append(_zscore(vals.astype(np.float64)))

        n = len(ids)
        if n < self.n_clusters:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n} time series")

        length = max(len(s) for s in series_list)

        # Pad series to same length
        padded = []
        for s in series_list:
            if len(s) < length:
                s = np.pad(s, (0, length - len(s)))
            padded.append(s)

        # Initialize assignments round-robin
        assignments = [i % self.n_clusters for i in range(n)]

        # Initialize centroids
        centroids = [np.zeros(length) for _ in range(self.n_clusters)]
        for ci in range(self.n_clusters):
            members = [padded[i] for i in range(n) if assignments[i] == ci]
            if members:
                centroids[ci] = _shape_extraction(members, length)

        # Iterate
        for _ in range(self.max_iter):
            # Assignment step
            new_assignments = []
            for i in range(n):
                best_cluster = 0
                best_dist = float("inf")
                for ci in range(self.n_clusters):
                    dist, _ = _sbd(padded[i], centroids[ci])
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = ci
                new_assignments.append(best_cluster)

            # Check convergence
            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centroids
            for ci in range(self.n_clusters):
                members = [padded[i] for i in range(n) if assignments[i] == ci]
                if members:
                    centroids[ci] = _shape_extraction(members, length)

        self.centroids_ = centroids
        self.labels_ = pl.DataFrame(
            {
                "unique_id": ids,
                "cluster": assignments,
            }
        )
        return self
