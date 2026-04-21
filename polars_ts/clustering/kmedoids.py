"""K-Medoids (PAM) clustering for time series using precomputed distances.

Delegates the PAM swap loop to Rust when available (10-30x faster),
falling back to pure Python otherwise.
"""

from __future__ import annotations

import random
from typing import Any

import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


class TimeSeriesKMedoids:
    """K-Medoids (PAM) time series clustering.

    Parameters
    ----------
    n_clusters
        Number of clusters. Default 2.
    metric
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``). Default ``"dtw"``.
    max_iter
        Maximum swap iterations. Default 100.
    seed
        Random seed for initial medoid selection. Default 42.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: str = "dtw",
        max_iter: int = 100,
        seed: int = 42,
        **distance_kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.seed = seed
        self.distance_kwargs = distance_kwargs
        self.labels_: pl.DataFrame | None = None
        self.medoids_: list[str] = []

    def fit(self, df: pl.DataFrame) -> TimeSeriesKMedoids:
        """Fit k-medoids clustering.

        Parameters
        ----------
        df
            DataFrame with columns ``unique_id`` and ``y``.

        Returns
        -------
        self

        """
        ids = df["unique_id"].unique().sort().to_list()
        n = len(ids)
        if self.n_clusters > n:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n} time series")

        result = kmedoids(
            df,
            k=self.n_clusters,
            method=self.metric,
            max_iter=self.max_iter,
            seed=self.seed,
            **self.distance_kwargs,
        )
        self.labels_ = result

        # Extract medoids from the PAM result
        dist_df = df.select(
            pl.col("unique_id").alias("unique_id"),
            pl.col("y").alias("y"),
        )
        pairwise = compute_distances(dist_df, dist_df, method=self.metric, **self.distance_kwargs)
        dist = pairwise_to_dict(pairwise)

        cluster_map: dict[int, list[str]] = {}
        for row in result.to_dicts():
            cid = row["cluster"]
            uid = str(row["unique_id"])
            cluster_map.setdefault(cid, []).append(uid)

        medoids = []
        for cid in sorted(cluster_map):
            members = cluster_map[cid]
            best_medoid = min(
                members,
                key=lambda m: sum(dist.get((m, o), 0.0) for o in members),
            )
            medoids.append(best_medoid)
        self.medoids_ = medoids

        return self


def _build_dist_matrix(dist_dict: dict[tuple[str, str], float], str_ids: list[str]) -> list[float]:
    """Convert distance dict to flat n×n row-major matrix."""
    n = len(str_ids)
    id_to_idx = {sid: i for i, sid in enumerate(str_ids)}
    flat = [0.0] * (n * n)
    for (a, b), d in dist_dict.items():
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            flat[i * n + j] = d
    return flat


def _kmedoids_rust(
    dist_dict: dict[tuple[str, str], float],
    str_ids: list[str],
    k: int,
    max_iter: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Run PAM via Rust extension."""
    from polars_ts_rs import kmedoids_pam

    n = len(str_ids)
    flat = _build_dist_matrix(dist_dict, str_ids)
    medoid_indices, assignments = kmedoids_pam(flat, n, k, max_iter, seed)
    return medoid_indices, assignments


def _kmedoids_python(
    dist_dict: dict[tuple[str, str], float],
    str_ids: list[str],
    k: int,
    max_iter: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Pure-Python PAM swap fallback."""
    dist = dist_dict

    # Self-distance is 0
    for sid in str_ids:
        dist[(sid, sid)] = 0.0

    def _total_cost(medoids: list[str]) -> float:
        return sum(min(dist[(sid, m)] for m in medoids) for sid in str_ids)

    def _assign(medoids: list[str]) -> dict[str, str]:
        return {sid: min(medoids, key=lambda m: dist[(sid, m)]) for sid in str_ids}

    # Initialize medoids randomly
    rng = random.Random(seed)
    medoid_names = rng.sample(str_ids, k)

    assignments_map = _assign(medoid_names)
    current_cost = _total_cost(medoid_names)

    # PAM swap loop
    for _ in range(max_iter):
        improved = False
        for i, _med in enumerate(medoid_names):
            non_medoids = [s for s in str_ids if s not in medoid_names]
            for candidate in non_medoids:
                new_medoids = medoid_names[:i] + [candidate] + medoid_names[i + 1 :]
                new_cost = _total_cost(new_medoids)
                if new_cost < current_cost - 1e-12:
                    medoid_names = new_medoids
                    assignments_map = _assign(medoid_names)
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Convert to index-based output
    sorted_medoid_names = sorted(medoid_names)
    medoid_to_label = {m: i for i, m in enumerate(sorted_medoid_names)}
    medoid_indices = [str_ids.index(m) for m in sorted_medoid_names]
    assignment_labels = [medoid_to_label[assignments_map[sid]] for sid in str_ids]
    return medoid_indices, assignment_labels


def kmedoids(
    df: pl.DataFrame,
    k: int,
    method: str = "dtw",
    max_iter: int = 100,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """K-Medoids (PAM) clustering over time series.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of clusters.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``).
    max_iter
        Maximum swap iterations.
    seed
        Random seed for initial medoid selection.
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
    ids = df[id_col].unique().sort().to_list()
    n = len(ids)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of series ({n})")

    # Compute pairwise distances
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist_dict = pairwise_to_dict(pairwise)

    str_ids = [str(i) for i in ids]

    # Self-distance is 0
    for sid in str_ids:
        dist_dict[(sid, sid)] = 0.0

    # Try Rust, fall back to Python
    try:
        _medoid_indices, assignment_labels = _kmedoids_rust(dist_dict, str_ids, k, max_iter, seed)
    except ImportError:
        _medoid_indices, assignment_labels = _kmedoids_python(dist_dict, str_ids, k, max_iter, seed)

    rows = list(zip(ids, assignment_labels, strict=False))
    return pl.DataFrame(
        {id_col: [r[0] for r in rows], "cluster": [r[1] for r in rows]},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )
