"""U-Shapelet (unsupervised shapelet) clustering for time series.

Discovers discriminative subsequences (shapelets) that separate groups of
time series, then clusters in shapelet-distance space.

References
----------
- Zakaria, J. et al. (2012). Clustering Time Series Using
  Unsupervised-Shapelets. ICDM.

"""

from __future__ import annotations

from typing import Self

import numpy as np
import polars as pl


def _extract_series(
    df: pl.DataFrame,
    target_col: str,
    id_col: str,
    time_col: str | None,
) -> tuple[list[str], np.ndarray]:
    """Extract series as a zero-padded 2-D array (n_series, max_len)."""
    sort_cols = [id_col, time_col] if time_col and time_col in df.columns else [id_col]
    sorted_df = df.sort(sort_cols)
    groups = sorted_df.group_by(id_col, maintain_order=True)
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    for key, group in groups:
        ids.append(key[0] if isinstance(key, tuple) else key)  # type: ignore[arg-type]
        arrays.append(group[target_col].to_numpy().astype(np.float64))

    max_len = max(a.shape[0] for a in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.float64)
    for i, a in enumerate(arrays):
        padded[i, : a.shape[0]] = a
    return ids, padded


def _subsequence_distance(shapelet: np.ndarray, series: np.ndarray) -> float:
    """Minimum sliding-window Euclidean distance between shapelet and series."""
    s_len = len(shapelet)
    t_len = len(series)
    if s_len > t_len:
        return float("inf")
    best = float("inf")
    for i in range(t_len - s_len + 1):
        d = 0.0
        for j in range(s_len):
            diff = shapelet[j] - series[i + j]
            d += diff * diff
            if d >= best:
                break
        if d < best:
            best = d
    return float(np.sqrt(best))


def _extract_candidates(
    X: np.ndarray,
    shapelet_lengths: list[int],
    n_candidates: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Extract random shapelet candidates from the dataset."""
    n_series, series_len = X.shape
    candidates: list[np.ndarray] = []
    for _ in range(n_candidates):
        s_len = int(rng.choice(shapelet_lengths))
        s_len = min(s_len, series_len)
        idx = rng.integers(0, n_series)
        start = rng.integers(0, max(1, series_len - s_len + 1))
        candidates.append(X[idx, start : start + s_len].copy())
    return candidates


def _score_shapelet(shapelet: np.ndarray, X: np.ndarray) -> float:
    """Score a shapelet candidate using the gap statistic.

    Computes the distances from the shapelet to all series, then finds
    the split point that maximizes the gap between successive sorted
    distances. A larger gap means the shapelet better separates series
    into two groups.
    """
    dists = np.array([_subsequence_distance(shapelet, X[i]) for i in range(X.shape[0])])
    sorted_dists = np.sort(dists)
    if len(sorted_dists) < 2:
        return 0.0
    gaps = np.diff(sorted_dists)
    return float(np.max(gaps))


def _kmeans_1d(
    distances: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> np.ndarray:
    """Run k-means on a distance-feature matrix."""
    n, d = distances.shape
    indices = rng.choice(n, size=min(k, n), replace=False)
    centroids = distances[indices].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign
        for i in range(n):
            best_c = 0
            best_d = float("inf")
            for c in range(len(centroids)):
                dist = float(np.sum((distances[i] - centroids[c]) ** 2))
                if dist < best_d:
                    best_d = dist
                    best_c = c
            labels[i] = best_c
        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(len(centroids), dtype=int)
        for i in range(n):
            new_centroids[labels[i]] += distances[i]
            counts[labels[i]] += 1
        changed = False
        for c in range(len(centroids)):
            if counts[c] > 0:
                nc = new_centroids[c] / counts[c]
                if not np.allclose(nc, centroids[c]):
                    changed = True
                centroids[c] = nc
        if not changed:
            break
    return labels


class UShapeletClusterer:
    """Unsupervised shapelet-based time series clustering.

    Discovers discriminative subsequences (shapelets) and clusters
    series by their shapelet distances.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    n_shapelets
        Number of shapelets to select.
    shapelet_lengths
        Candidate shapelet lengths to consider.
    n_candidates
        Number of random shapelet candidates to evaluate.
    target_col
        Column with the values to cluster.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    seed
        Random seed for reproducibility.
    max_iter
        Maximum k-means iterations.

    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_shapelets: int = 10,
        shapelet_lengths: list[int] | None = None,
        n_candidates: int = 100,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
        seed: int = 42,
        max_iter: int = 100,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_shapelets = n_shapelets
        self.shapelet_lengths = shapelet_lengths or [10, 20, 30]
        self.n_candidates = n_candidates
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.seed = seed
        self.max_iter = max_iter
        self.labels_: pl.DataFrame | None = None
        self.shapelets_: list[np.ndarray] = []

    def fit(self, df: pl.DataFrame) -> Self:
        """Discover shapelets and cluster time series.

        Parameters
        ----------
        df
            Input DataFrame with time series data.

        Returns
        -------
        Self

        """
        ids, X = _extract_series(df, self.target_col, self.id_col, self.time_col)
        n_series = X.shape[0]

        if self.n_clusters > n_series:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n_series} series")

        rng = np.random.default_rng(self.seed)

        # Clamp shapelet lengths to series length
        series_len = X.shape[1]
        valid_lengths = [min(sl, series_len) for sl in self.shapelet_lengths]

        # Extract and score candidates
        candidates = _extract_candidates(X, valid_lengths, self.n_candidates, rng)
        scores = [(i, _score_shapelet(c, X)) for i, c in enumerate(candidates)]
        scores.sort(key=lambda t: t[1], reverse=True)

        # Select top shapelets
        n_select = min(self.n_shapelets, len(candidates))
        self.shapelets_ = [candidates[scores[i][0]] for i in range(n_select)]

        # Build distance matrix (n_series, n_shapelets)
        dist_matrix = np.empty((n_series, n_select), dtype=np.float64)
        for s_idx, shapelet in enumerate(self.shapelets_):
            for t_idx in range(n_series):
                dist_matrix[t_idx, s_idx] = _subsequence_distance(shapelet, X[t_idx])

        # Cluster in distance space
        labels = _kmeans_1d(dist_matrix, self.n_clusters, rng, self.max_iter)

        self.labels_ = pl.DataFrame({self.id_col: ids, "cluster": labels.tolist()})
        return self


def shapelet_cluster(
    df: pl.DataFrame,
    k: int = 3,
    n_shapelets: int = 10,
    shapelet_lengths: list[int] | None = None,
    n_candidates: int = 100,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    seed: int = 42,
    max_iter: int = 100,
) -> pl.DataFrame:
    """Discover U-Shapelets and cluster time series.

    Convenience function wrapping :class:`UShapeletClusterer`.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    k
        Number of clusters.
    n_shapelets
        Number of shapelets to select.
    shapelet_lengths
        Candidate shapelet lengths to consider.
    n_candidates
        Number of random shapelet candidates to evaluate.
    target_col
        Column with the values to transform.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    seed
        Random seed for reproducibility.
    max_iter
        Maximum k-means iterations.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    usc = UShapeletClusterer(
        n_clusters=k,
        n_shapelets=n_shapelets,
        shapelet_lengths=shapelet_lengths,
        n_candidates=n_candidates,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        seed=seed,
        max_iter=max_iter,
    )
    usc.fit(df)
    assert usc.labels_ is not None
    return usc.labels_
