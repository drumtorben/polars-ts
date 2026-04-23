"""Agglomerative (hierarchical) clustering for time series.

Computes pairwise distances via the existing Rust-accelerated distance
engine, converts to a condensed distance matrix, and delegates to
``scipy.cluster.hierarchy`` for linkage and tree cutting.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from polars_ts.clustering.density import _compute_distance_matrix

_VALID_LINKAGE = {"single", "complete", "average", "weighted"}


@overload
def agglomerative_cluster(
    df: pl.DataFrame,
    method: str = ...,
    n_clusters: int = ...,
    linkage_method: str = ...,
    id_col: str = ...,
    target_col: str = ...,
    *,
    return_linkage: bool = False,
    **distance_kwargs: Any,
) -> pl.DataFrame: ...


@overload
def agglomerative_cluster(
    df: pl.DataFrame,
    method: str = ...,
    n_clusters: int = ...,
    linkage_method: str = ...,
    id_col: str = ...,
    target_col: str = ...,
    *,
    return_linkage: bool = True,
    **distance_kwargs: Any,
) -> tuple[pl.DataFrame, np.ndarray]: ...


def agglomerative_cluster(
    df: pl.DataFrame,
    method: str = "dtw",
    n_clusters: int = 2,
    linkage_method: str = "average",
    id_col: str = "unique_id",
    target_col: str = "y",
    *,
    return_linkage: bool = False,
    **distance_kwargs: Any,
) -> pl.DataFrame | tuple[pl.DataFrame, np.ndarray]:
    """Agglomerative (hierarchical) clustering over time series.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``).
    n_clusters
        Number of clusters to produce.
    linkage_method
        Linkage criterion: ``"single"``, ``"complete"``, ``"average"``,
        or ``"weighted"``.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    return_linkage
        If *True*, also return the linkage matrix (compatible with
        ``scipy.cluster.hierarchy.dendrogram``).
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    pl.DataFrame or (pl.DataFrame, np.ndarray)
        DataFrame with columns ``[id_col, "cluster"]``.
        When ``return_linkage=True``, a tuple of ``(labels, linkage_matrix)``
        is returned.

    """
    ids = df[id_col].unique().sort().to_list()
    n = len(ids)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > n:
        raise ValueError(f"n_clusters ({n_clusters}) must be <= number of series ({n})")
    if linkage_method not in _VALID_LINKAGE:
        raise ValueError(f"Unknown linkage {linkage_method!r}. Valid: {sorted(_VALID_LINKAGE)}")

    _, dist_mat = _compute_distance_matrix(df, method, id_col, target_col, **distance_kwargs)

    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method=linkage_method)
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # fcluster labels start at 1; convert to 0-based
    labels_0 = (cluster_labels - 1).tolist()

    result = pl.DataFrame(
        {id_col: ids, "cluster": labels_0},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )

    if return_linkage:
        return result, Z
    return result
