"""k-Medoids clustering (PAM) for time series using precomputed distance matrices."""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_ts.distance import compute_pairwise_distance


class TimeSeriesKMedoids:
    """k-Medoids (PAM) clustering for time series.

    Computes a full pairwise distance matrix using the unified distance API,
    then applies the PAM (Partitioning Around Medoids) algorithm to find
    cluster assignments.

    Args:
        n_clusters: Number of clusters.
        metric: Distance metric name. Default ``"dtw"``.
        max_iter: Maximum number of PAM iterations. Default 100.
        **metric_kwargs: Additional keyword arguments passed to the distance function.

    Examples:
        >>> km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        >>> result = km.fit(df)
        >>> result.labels_

    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: str = "dtw",
        max_iter: int = 100,
        **metric_kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.metric_kwargs = metric_kwargs
        self.labels_: pl.DataFrame | None = None
        self.medoids_: list[str] = []
        self._dist_matrix: dict[tuple[str, str], float] = {}

    def fit(self, df: pl.DataFrame) -> TimeSeriesKMedoids:
        """Fit k-Medoids clustering.

        Args:
            df: DataFrame with columns ``unique_id`` and ``y``.

        Returns:
            self, with ``labels_`` and ``medoids_`` populated.

        """
        ts_df = df.select("unique_id", "y")
        ids = sorted(ts_df["unique_id"].unique().cast(pl.String).to_list())
        n = len(ids)

        if n < self.n_clusters:
            raise ValueError(
                f"Cannot create {self.n_clusters} clusters from {n} time series"
            )

        # Compute pairwise distance matrix
        distances = compute_pairwise_distance(
            ts_df, ts_df, method=self.metric, **self.metric_kwargs
        )
        dist_col = [c for c in distances.columns if c not in ("id_1", "id_2")][0]

        # Build symmetric distance lookup
        dist: dict[tuple[str, str], float] = {}
        for row in distances.to_dicts():
            id1 = str(row["id_1"])
            id2 = str(row["id_2"])
            d = row[dist_col]
            dist[(id1, id2)] = d
            dist[(id2, id1)] = d
        for uid in ids:
            dist[(uid, uid)] = 0.0
        self._dist_matrix = dist

        # Initialize medoids: pick first n_clusters ids (deterministic)
        medoids = ids[: self.n_clusters]

        # PAM swap phase
        for _ in range(self.max_iter):
            # Assign each point to nearest medoid
            assignments = self._assign(ids, medoids)

            # Try swapping each medoid with each non-medoid
            improved = False
            best_cost = self._total_cost(ids, medoids)
            for mi, m in enumerate(medoids):
                for candidate in ids:
                    if candidate in medoids:
                        continue
                    new_medoids = medoids.copy()
                    new_medoids[mi] = candidate
                    cost = self._total_cost(ids, new_medoids)
                    if cost < best_cost:
                        best_cost = cost
                        medoids = new_medoids
                        improved = True

            if not improved:
                break

        # Final assignment
        assignments = self._assign(ids, medoids)
        self.medoids_ = medoids
        self.labels_ = pl.DataFrame({
            "unique_id": ids,
            "cluster": [assignments[uid] for uid in ids],
        })
        return self

    def _assign(self, ids: list[str], medoids: list[str]) -> dict[str, int]:
        """Assign each id to its nearest medoid."""
        assignments: dict[str, int] = {}
        for uid in ids:
            best_cluster = 0
            best_dist = float("inf")
            for ci, m in enumerate(medoids):
                d = self._dist_matrix.get((uid, m), float("inf"))
                if d < best_dist:
                    best_dist = d
                    best_cluster = ci
            assignments[uid] = best_cluster
        return assignments

    def _total_cost(self, ids: list[str], medoids: list[str]) -> float:
        """Compute total cost (sum of distances to nearest medoid)."""
        total = 0.0
        for uid in ids:
            min_dist = float("inf")
            for m in medoids:
                d = self._dist_matrix.get((uid, m), float("inf"))
                if d < min_dist:
                    min_dist = d
            total += min_dist
        return total
