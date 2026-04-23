"""Density-based clustering (HDBSCAN / DBSCAN) for time series.

Computes pairwise distances via the existing Rust-accelerated distance
engine and passes the precomputed matrix to scikit-learn's implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


def _build_square_matrix(
    dist_dict: dict[tuple[str, str], float],
    ids: list[str],
) -> np.ndarray:
    """Convert symmetric distance dict to a square numpy matrix."""
    n = len(ids)
    idx = {uid: i for i, uid in enumerate(ids)}
    mat = np.zeros((n, n), dtype=np.float64)
    for (a, b), d in dist_dict.items():
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = d
    return mat


def hdbscan_cluster(
    df: pl.DataFrame,
    method: str = "dtw",
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """HDBSCAN clustering over time series using precomputed distances.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``).
    min_cluster_size
        Minimum cluster size for HDBSCAN.
    min_samples
        Number of samples in a neighbourhood for a point to be a core point.
        Defaults to ``min_cluster_size`` when *None*.
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
        Noise points are labelled ``-1``.

    """
    from sklearn.cluster import HDBSCAN

    ids, dist_mat = _compute_distance_matrix(df, method, id_col, target_col, **distance_kwargs)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_mat)

    raw_ids = df[id_col].unique().sort().to_list()
    return pl.DataFrame(
        {id_col: raw_ids, "cluster": labels.tolist()},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )


def dbscan_cluster(
    df: pl.DataFrame,
    method: str = "dtw",
    eps: float = 0.5,
    min_samples: int = 5,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """DBSCAN clustering over time series using precomputed distances.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``).
    eps
        Maximum distance between two samples in the same neighbourhood.
    min_samples
        Number of samples in a neighbourhood for a point to be a core point.
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
        Noise points are labelled ``-1``.

    """
    from sklearn.cluster import DBSCAN

    ids, dist_mat = _compute_distance_matrix(df, method, id_col, target_col, **distance_kwargs)

    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_mat)

    raw_ids = df[id_col].unique().sort().to_list()
    return pl.DataFrame(
        {id_col: raw_ids, "cluster": labels.tolist()},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )


def _compute_distance_matrix(
    df: pl.DataFrame,
    method: str,
    id_col: str,
    target_col: str,
    **distance_kwargs: Any,
) -> tuple[list[str], np.ndarray]:
    """Shared helper: compute pairwise distances and return (sorted ids, square matrix)."""
    ids = [str(i) for i in df[id_col].unique().sort().to_list()]
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist_dict = pairwise_to_dict(pairwise)

    # Ensure self-distances are zero
    for sid in ids:
        dist_dict[(sid, sid)] = 0.0

    mat = _build_square_matrix(dist_dict, ids)
    return ids, mat
