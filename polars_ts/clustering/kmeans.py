"""K-Means clustering for time series with DTW Barycentric Averaging (DBA).

Uses DTW-based distances for the assignment step and DBA for the centroid
update step, producing synthetic centroids that better represent cluster
averages than medoid-based approaches.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import polars as pl

from polars_ts.clustering.dba import dba


class TimeSeriesKMeans:
    """K-Means clustering for time series using DBA centroids.

    Parameters
    ----------
    n_clusters
        Number of clusters. Default 2.
    metric
        Distance metric name. Currently only ``"dtw"`` is supported
        (DBA requires DTW alignment paths). Default ``"dtw"``.
    max_iter
        Maximum number of k-means iterations. Default 50.
    dba_max_iter
        Maximum DBA refinement iterations per centroid update. Default 30.
    seed
        Random seed for initial centroid selection. Default 42.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: str = "dtw",
        max_iter: int = 50,
        dba_max_iter: int = 30,
        seed: int = 42,
        **distance_kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.dba_max_iter = dba_max_iter
        self.seed = seed
        self.distance_kwargs = distance_kwargs
        self.labels_: pl.DataFrame | None = None
        self.centroids_: list[np.ndarray] = []

    def fit(
        self,
        df: pl.DataFrame,
        id_col: str = "unique_id",
        target_col: str = "y",
    ) -> TimeSeriesKMeans:
        """Fit k-means clustering with DBA centroids.

        Parameters
        ----------
        df
            DataFrame with columns ``id_col`` and ``target_col``.
        id_col
            Column identifying each time series.
        target_col
            Column with the time series values.

        Returns
        -------
        self

        """
        if self.metric != "dtw":
            raise ValueError(f"Only metric='dtw' is supported for DBA-based k-means, got {self.metric!r}")

        id_dtype = df[id_col].dtype
        ids = sorted(df[id_col].unique().cast(pl.String).to_list())
        n = len(ids)
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.n_clusters > n:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n} time series")

        # Extract series as numpy arrays
        series_map: dict[str, np.ndarray] = {}
        for uid in ids:
            vals = df.filter(pl.col(id_col).cast(pl.String) == uid)[target_col].to_numpy()
            series_map[uid] = vals.astype(np.float64)

        series_list = [series_map[uid] for uid in ids]

        # Initialize centroids by picking k random series
        rng = random.Random(self.seed)
        init_indices = rng.sample(range(n), self.n_clusters)
        centroids = [series_list[i].copy() for i in init_indices]

        assignments = [-1] * n

        for _ in range(self.max_iter):
            # Assignment step: assign each series to nearest centroid
            new_assignments = self._assign(series_list, centroids)

            # Check convergence
            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centroids via DBA
            centroids = self._update_centroids(series_list, assignments)

        self.centroids_ = centroids
        self.labels_ = pl.DataFrame(
            {id_col: ids, "cluster": assignments},
            schema={id_col: pl.String, "cluster": pl.Int64},
        ).with_columns(pl.col(id_col).cast(id_dtype))
        return self

    def _assign(
        self,
        series_list: list[np.ndarray],
        centroids: list[np.ndarray],
    ) -> list[int]:
        """Assign each series to the nearest centroid using DTW distance."""
        assignments = []
        for s in series_list:
            best_cluster = 0
            best_dist = float("inf")
            for ci, c in enumerate(centroids):
                d = self._dtw_distance(s, c)
                if d < best_dist:
                    best_dist = d
                    best_cluster = ci
            assignments.append(best_cluster)
        return assignments

    @staticmethod
    def _dtw_distance(s: np.ndarray, t: np.ndarray) -> float:
        """Compute DTW distance between two series."""
        n, m = len(s), len(t)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = (s[i - 1] - t[j - 1]) ** 2
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[n, m])

    def _update_centroids(
        self,
        series_list: list[np.ndarray],
        assignments: list[int],
    ) -> list[np.ndarray]:
        """Recompute centroids via DBA."""
        centroids = []
        for ci in range(self.n_clusters):
            members = [series_list[i] for i, a in enumerate(assignments) if a == ci]
            if members:
                centroids.append(dba(members, max_iter=self.dba_max_iter))
            else:
                # Empty cluster: reinitialize with a random series
                centroids.append(series_list[random.Random(self.seed + ci).randint(0, len(series_list) - 1)].copy())
        return centroids


def kmeans_dba(
    df: pl.DataFrame,
    k: int,
    method: str = "dtw",
    max_iter: int = 50,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """K-Means clustering with DBA centroids (convenience function).

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of clusters.
    method
        Distance metric name (e.g. ``"dtw"``). Default ``"dtw"``.
    max_iter
        Maximum k-means iterations.
    seed
        Random seed for reproducibility.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    km = TimeSeriesKMeans(
        n_clusters=k,
        metric=method,
        max_iter=max_iter,
        seed=seed,
        **distance_kwargs,
    )
    km.fit(df, id_col=id_col, target_col=target_col)
    assert km.labels_ is not None
    return km.labels_
