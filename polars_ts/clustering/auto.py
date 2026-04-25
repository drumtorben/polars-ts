"""Automated clustering pipeline selection for time series.

Performs grid search over method x distance x k combinations and returns
the best result according to the chosen evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class AutoClusterResult:
    """Result of an ``auto_cluster`` search.

    Attributes
    ----------
    best_labels
        DataFrame with ``[id_col, "cluster"]`` for the best combination.
    best_method
        Name of the winning clustering method.
    best_distance
        Name of the winning distance metric.
    best_k
        Winning k (``None`` for density-based methods).
    best_score
        Evaluation score of the winning combination.
    results_table
        DataFrame summarising every evaluated combination with columns
        ``["method", "distance", "k", "score"]``.

    """

    best_labels: pl.DataFrame
    best_method: str
    best_distance: str
    best_k: int | None
    best_score: float
    results_table: pl.DataFrame


# Methods that accept a k parameter
_K_METHODS = {"kmedoids", "kshape", "spectral"}
# Methods that ignore k (density-based)
_NO_K_METHODS = {"hdbscan", "dbscan"}
# Higher-is-better metrics
_HIGHER_IS_BETTER = {"silhouette", "calinski_harabasz"}


def _run_clustering(
    df: pl.DataFrame,
    method: str,
    distance: str,
    k: int | None,
    id_col: str,
    target_col: str,
    seed: int,
    hdbscan_kwargs: dict[str, Any],
    dbscan_kwargs: dict[str, Any],
) -> pl.DataFrame | None:
    """Run a single clustering method and return labels or None on failure."""
    try:
        if method == "kmedoids":
            from polars_ts.clustering.kmedoids import kmedoids

            assert k is not None
            return kmedoids(df, k=k, method=distance, id_col=id_col, target_col=target_col, seed=seed)

        if method == "spectral":
            from polars_ts.clustering.spectral import spectral_cluster

            assert k is not None
            return spectral_cluster(df, k=k, method=distance, id_col=id_col, target_col=target_col, seed=seed)

        if method == "kshape":
            from polars_ts.clustering.kshape import KShape

            assert k is not None
            ks_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
            ks = KShape(n_clusters=k).fit(ks_df)
            labels: pl.DataFrame = ks.labels_
            if id_col != "unique_id":
                id_map = dict(
                    zip(
                        df.select(pl.col(id_col).cast(pl.String)).to_series().unique().sort().to_list(),
                        df[id_col].unique().sort().to_list(),
                        strict=False,
                    )
                )
                labels = labels.with_columns(pl.col("unique_id").replace_strict(id_map).alias(id_col)).drop("unique_id")
            return labels

        if method == "hdbscan":
            from polars_ts.clustering.density import hdbscan_cluster

            return hdbscan_cluster(df, method=distance, id_col=id_col, target_col=target_col, **hdbscan_kwargs)

        if method == "dbscan":
            from polars_ts.clustering.density import dbscan_cluster

            return dbscan_cluster(df, method=distance, id_col=id_col, target_col=target_col, **dbscan_kwargs)

    except Exception:
        return None

    return None


def _evaluate(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    distance: str,
    metric: str,
    id_col: str,
    target_col: str,
) -> float | None:
    """Evaluate clustering quality. Returns None if evaluation fails."""
    from polars_ts.clustering.evaluation import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_clusters = labels["cluster"].n_unique()
    # Need at least 2 clusters for meaningful evaluation
    if n_clusters < 2:
        return None

    try:
        if metric == "silhouette":
            return silhouette_score(df, labels, method=distance, id_col=id_col, target_col=target_col)
        if metric == "davies_bouldin":
            return davies_bouldin_score(df, labels, method=distance, id_col=id_col, target_col=target_col)
        if metric == "calinski_harabasz":
            return calinski_harabasz_score(df, labels, method=distance, id_col=id_col, target_col=target_col)
    except Exception:
        return None

    return None


def auto_cluster(
    df: pl.DataFrame,
    methods: list[str] | None = None,
    distances: list[str] | None = None,
    k_range: range | None = None,
    metric: str = "silhouette",
    id_col: str = "unique_id",
    target_col: str = "y",
    seed: int = 42,
    hdbscan_kwargs: dict[str, Any] | None = None,
    dbscan_kwargs: dict[str, Any] | None = None,
) -> AutoClusterResult:
    """Automated clustering pipeline selection via grid search.

    Enumerates method x distance x k combinations, evaluates each with the
    chosen metric, and returns the best result.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.

    Methods
    -------
        Clustering methods to try. Default ``["kmedoids", "spectral"]``.
    distances
        Distance metrics to try. Default ``["sbd", "dtw"]``.
    k_range
        Range of k values for methods that accept k. Default ``range(2, 6)``.
    metric
        Evaluation metric: ``"silhouette"`` (higher=better),
        ``"davies_bouldin"`` (lower=better), or ``"calinski_harabasz"``
        (higher=better).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    seed
        Random seed for clustering methods.
    hdbscan_kwargs
        Extra keyword arguments for HDBSCAN (e.g. ``min_cluster_size``).
    dbscan_kwargs
        Extra keyword arguments for DBSCAN (e.g. ``eps``, ``min_samples``).

    Returns
    -------
    AutoClusterResult
        Structured result with best labels, metadata, and results table.

    Raises
    ------
    ValueError
        If ``metric`` is unknown or no valid combinations are found.

    """
    valid_metrics = {"silhouette", "davies_bouldin", "calinski_harabasz"}
    if metric not in valid_metrics:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(valid_metrics)}")

    if methods is None:
        methods = ["kmedoids", "spectral"]
    if distances is None:
        distances = ["sbd", "dtw"]
    if k_range is None:
        k_range = range(2, 6)
    if hdbscan_kwargs is None:
        hdbscan_kwargs = {}
    if dbscan_kwargs is None:
        dbscan_kwargs = {}

    higher_is_better = metric in _HIGHER_IS_BETTER

    results: list[dict[str, Any]] = []
    best_labels: pl.DataFrame | None = None
    best_score: float | None = None

    for method in methods:
        for distance in distances:
            # kshape only works with SBD
            if method == "kshape" and distance != "sbd":
                continue

            if method in _NO_K_METHODS:
                # Density-based: run once per distance (no k)
                labels = _run_clustering(
                    df,
                    method,
                    distance,
                    k=None,
                    id_col=id_col,
                    target_col=target_col,
                    seed=seed,
                    hdbscan_kwargs=hdbscan_kwargs,
                    dbscan_kwargs=dbscan_kwargs,
                )
                if labels is None:
                    continue

                score = _evaluate(df, labels, distance, metric, id_col, target_col)
                if score is None:
                    continue

                results.append({"method": method, "distance": distance, "k": None, "score": score})

                if best_score is None or (
                    (higher_is_better and score > best_score) or (not higher_is_better and score < best_score)
                ):
                    best_score = score
                    best_labels = labels
            else:
                # k-based methods
                for k in k_range:
                    labels = _run_clustering(
                        df,
                        method,
                        distance,
                        k=k,
                        id_col=id_col,
                        target_col=target_col,
                        seed=seed,
                        hdbscan_kwargs=hdbscan_kwargs,
                        dbscan_kwargs=dbscan_kwargs,
                    )
                    if labels is None:
                        continue

                    score = _evaluate(df, labels, distance, metric, id_col, target_col)
                    if score is None:
                        continue

                    results.append({"method": method, "distance": distance, "k": k, "score": score})

                    if best_score is None or (
                        (higher_is_better and score > best_score) or (not higher_is_better and score < best_score)
                    ):
                        best_score = score
                        best_labels = labels

    if not results or best_labels is None or best_score is None:
        raise ValueError("No valid clustering combinations were found")

    # Find the best row
    best_row = results[0]
    for row in results[1:]:
        if higher_is_better and row["score"] > best_row["score"]:
            best_row = row
        elif not higher_is_better and row["score"] < best_row["score"]:
            best_row = row

    results_table = pl.DataFrame(
        results,
        schema={"method": pl.String, "distance": pl.String, "k": pl.Int64, "score": pl.Float64},
    )

    return AutoClusterResult(
        best_labels=best_labels,
        best_method=best_row["method"],
        best_distance=best_row["distance"],
        best_k=best_row["k"],
        best_score=best_row["score"],
        results_table=results_table,
    )
