"""Spectral clustering (K-Spectral Centroid) for time series.

Computes pairwise distances via the existing Rust-accelerated distance
engine, converts to an affinity matrix using a Gaussian kernel, and
clusters on the eigenvectors of the graph Laplacian.
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


def _compute_distance_matrix(
    df: pl.DataFrame,
    method: str,
    id_col: str,
    target_col: str,
    **distance_kwargs: Any,
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise distances and return (sorted ids, square matrix)."""
    ids = [str(i) for i in df[id_col].unique().sort().to_list()]
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist_dict = pairwise_to_dict(pairwise)

    for sid in ids:
        dist_dict[(sid, sid)] = 0.0

    mat = _build_square_matrix(dist_dict, ids)
    return ids, mat


def spectral_cluster(
    df: pl.DataFrame,
    k: int = 3,
    method: str = "sbd",
    sigma: float = 1.0,
    id_col: str = "unique_id",
    target_col: str = "y",
    seed: int = 42,
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """Spectral clustering over time series using a kernel affinity matrix.

    Builds a graph Laplacian from pairwise distances converted via a
    Gaussian kernel, then clusters the leading eigenvectors with k-means
    (K-Spectral Centroid approach).

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of clusters.
    method
        Distance metric name (e.g. ``"sbd"``, ``"dtw"``).
    sigma
        Bandwidth parameter for the Gaussian kernel:
        ``affinity[i,j] = exp(-dist[i,j]^2 / (2 * sigma^2))``.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    seed
        Random seed for k-means initialisation.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    from scipy.linalg import eigh
    from sklearn.cluster import KMeans

    ids, dist_mat = _compute_distance_matrix(df, method, id_col, target_col, **distance_kwargs)
    n = len(ids)

    if n < k:
        raise ValueError(f"Cannot create {k} clusters from {n} time series")

    # Gaussian kernel affinity
    affinity = np.exp(-(dist_mat**2) / (2.0 * sigma**2))

    # Normalized graph Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    degree = affinity.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    laplacian = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt

    # k smallest eigenvectors (eigenvalues in ascending order)
    eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, k - 1])
    embedding = eigenvectors[:, :k]

    # Row-normalise the embedding
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embedding = embedding / norms

    # k-means on the spectral embedding
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embedding)

    raw_ids = df[id_col].unique().sort().to_list()
    return pl.DataFrame(
        {id_col: raw_ids, "cluster": labels.tolist()},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )
