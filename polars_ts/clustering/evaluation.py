"""Clustering evaluation metrics for time series using precomputed distances."""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


def _build_dist_matrix(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    method: str,
    id_col: str,
    target_col: str,
    **distance_kwargs: Any,
) -> tuple[list[str], dict[str, int], dict[tuple[str, str], float]]:
    """Compute distance matrix and cluster assignments.

    Returns (ids, id_to_cluster, dist_dict).
    """
    dist_df = df.select(
        pl.col(id_col).alias("unique_id"),
        pl.col(target_col).alias("y"),
    )
    pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist = pairwise_to_dict(pairwise)

    ids = sorted(labels[id_col].cast(pl.String).to_list())
    id_to_cluster: dict[str, int] = {}
    for row in labels.to_dicts():
        id_to_cluster[str(row[id_col])] = int(row["cluster"])

    # Self-distance is 0
    for sid in ids:
        dist[(sid, sid)] = 0.0

    return ids, id_to_cluster, dist


def _group_by_cluster(id_to_cluster: dict[str, int]) -> dict[int, list[str]]:
    """Group series IDs by their cluster assignment."""
    cluster_members: dict[int, list[str]] = {}
    for sid, cid in id_to_cluster.items():
        cluster_members.setdefault(cid, []).append(sid)
    return cluster_members


def _find_medoids(
    cluster_members: dict[int, list[str]],
    dist: dict[tuple[str, str], float],
) -> dict[int, str]:
    """Find the medoid (member with min total distance) for each cluster."""
    medoids: dict[int, str] = {}
    for cid, members in cluster_members.items():
        medoids[cid] = min(
            members,
            key=lambda m: sum(dist[(m, o)] for o in members),
        )
    return medoids


def silhouette_score(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    method: str = "dtw",
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> float:
    """Compute the mean silhouette score for a clustering result.

    The silhouette score for each sample measures how similar it is to its own
    cluster compared to the nearest other cluster. Values range from -1 to 1,
    where higher is better.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    labels
        DataFrame with columns ``id_col`` and ``"cluster"`` (e.g. from ``kmedoids``).
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    float
        Mean silhouette score across all samples. Returns 0.0 if there is
        only one cluster or one sample.

    """
    samples = silhouette_samples(df, labels, method, id_col, target_col, **distance_kwargs)
    values = samples["silhouette"].to_list()
    if not values:
        return 0.0
    return sum(values) / len(values)


def silhouette_samples(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    method: str = "dtw",
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """Compute the silhouette score for each individual sample.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    labels
        DataFrame with columns ``id_col`` and ``"cluster"``.
    method
        Distance metric name.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``id_col``, ``"cluster"``, and ``"silhouette"``.

    """
    ids, id_to_cluster, dist = _build_dist_matrix(df, labels, method, id_col, target_col, **distance_kwargs)

    clusters = set(id_to_cluster.values())

    cluster_members = _group_by_cluster(id_to_cluster)

    rows: list[dict[str, Any]] = []
    for sid in ids:
        own_cluster = id_to_cluster[sid]
        own_members = cluster_members[own_cluster]

        if len(clusters) <= 1 or len(ids) <= 1:
            rows.append({id_col: sid, "cluster": own_cluster, "silhouette": 0.0})
            continue

        if len(own_members) <= 1:
            a_i = 0.0
        else:
            a_i = sum(dist[(sid, o)] for o in own_members if o != sid) / (len(own_members) - 1)

        b_i = float("inf")
        for cid, members in cluster_members.items():
            if cid == own_cluster:
                continue
            mean_dist = sum(dist[(sid, o)] for o in members) / len(members)
            b_i = min(b_i, mean_dist)

        if b_i == float("inf"):
            sil = 0.0
        else:
            denom = max(a_i, b_i)
            sil = (b_i - a_i) / denom if denom > 0 else 0.0

        rows.append({id_col: sid, "cluster": own_cluster, "silhouette": sil})

    return pl.DataFrame(rows)


def davies_bouldin_score(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    method: str = "dtw",
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> float:
    """Compute the Davies-Bouldin index for a clustering result.

    Lower values indicate better clustering. The index measures the average
    similarity between each cluster and its most similar cluster, where
    similarity is the ratio of within-cluster distances to between-cluster
    distances. Uses medoids instead of centroids, which is standard for
    non-Euclidean distance metrics.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    labels
        DataFrame with columns ``id_col`` and ``"cluster"``.
    method
        Distance metric name.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    float
        Davies-Bouldin index. Returns 0.0 if there is only one cluster.

    """
    ids, id_to_cluster, dist = _build_dist_matrix(df, labels, method, id_col, target_col, **distance_kwargs)

    cluster_members = _group_by_cluster(id_to_cluster)
    cluster_ids = sorted(cluster_members.keys())
    k = len(cluster_ids)
    if k <= 1:
        return 0.0

    medoids = _find_medoids(cluster_members, dist)

    # S_i: mean distance of members to their medoid
    scatter: dict[int, float] = {}
    for cid, members in cluster_members.items():
        med = medoids[cid]
        if len(members) <= 1:
            scatter[cid] = 0.0
        else:
            scatter[cid] = sum(dist[(m, med)] for m in members) / len(members)

    # R_ij = (S_i + S_j) / d(medoid_i, medoid_j)
    db_sum = 0.0
    for i_idx, ci in enumerate(cluster_ids):
        max_r = 0.0
        for j_idx, cj in enumerate(cluster_ids):
            if i_idx == j_idx:
                continue
            d_ij = dist[(medoids[ci], medoids[cj])]
            if d_ij == 0:
                r_ij = float("inf")
            else:
                r_ij = (scatter[ci] + scatter[cj]) / d_ij
            max_r = max(max_r, r_ij)
        db_sum += max_r

    return db_sum / k


def calinski_harabasz_score(
    df: pl.DataFrame,
    labels: pl.DataFrame,
    method: str = "dtw",
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> float:
    """Compute the Calinski-Harabasz index for a clustering result.

    Higher values indicate better-defined clusters. The index is the ratio
    of between-cluster dispersion to within-cluster dispersion, adjusted
    for the number of clusters and samples.

    .. note::
        This is a medoid-based adaptation of the standard Calinski-Harabasz
        index (which assumes Euclidean centroids). Results are meaningful for
        comparing clusterings under the same metric but may not be directly
        comparable to Euclidean implementations.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    labels
        DataFrame with columns ``id_col`` and ``"cluster"``.
    method
        Distance metric name.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    float
        Calinski-Harabasz index. Returns 0.0 if there is only one cluster
        or fewer than ``k + 1`` samples.

    """
    ids, id_to_cluster, dist = _build_dist_matrix(df, labels, method, id_col, target_col, **distance_kwargs)

    cluster_members = _group_by_cluster(id_to_cluster)
    cluster_ids = sorted(cluster_members.keys())
    k = len(cluster_ids)
    n = len(ids)

    if k <= 1 or n <= k:
        return 0.0

    medoids = _find_medoids(cluster_members, dist)

    # Global medoid: the series with min total distance to all others
    global_medoid = min(ids, key=lambda s: sum(dist[(s, o)] for o in ids))

    # Within-cluster dispersion: sum of squared distances to cluster medoid
    wk = 0.0
    for cid, members in cluster_members.items():
        med = medoids[cid]
        for m in members:
            d = dist[(m, med)]
            wk += d * d

    # Between-cluster dispersion: sum of n_i * d(medoid_i, global_medoid)^2
    bk = 0.0
    for cid, members in cluster_members.items():
        d = dist[(medoids[cid], global_medoid)]
        bk += len(members) * d * d

    if wk == 0:
        return 0.0

    return (bk / (k - 1)) / (wk / (n - k))
