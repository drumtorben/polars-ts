"""Scalable k-medoids variants: CLARA and CLARANS.

CLARA (Clustering LARge Applications) — subsample → PAM → repeat, keep best.
CLARANS (Clustering Large Applications based on RANdomized Search) —
randomized medoid neighborhood search over the full dataset.

References
----------
Kaufman, L. & Rousseeuw, P.J. (1990). *Finding Groups in Data*. Wiley.
Ng, R. & Han, J. (2002). *CLARANS: A method for clustering objects for
spatial data mining*. IEEE TKDE.

"""

from __future__ import annotations

import random
from typing import Any

import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict
from polars_ts.clustering.kmedoids import kmedoids


def clara(
    df: pl.DataFrame,
    k: int,
    method: str = "dtw",
    n_samples: int = 5,
    sample_size: int = 40,
    max_iter: int = 100,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """CLARA: subsample-based PAM for large datasets.

    Runs PAM on ``n_samples`` random subsamples of size ``sample_size``,
    evaluates the full-dataset cost for each, and returns the best result.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of clusters.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``). Default ``"dtw"``.
    n_samples
        Number of subsampling iterations. Default 5.
    sample_size
        Number of series per subsample. Clamped to the total number of
        series if larger. Default 40.
    max_iter
        Maximum PAM swap iterations per subsample. Default 100.
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
    ids = df[id_col].unique().sort().to_list()
    n = len(ids)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of series ({n})")

    # Clamp sample_size
    effective_sample = min(sample_size, n)
    if effective_sample < k:
        effective_sample = n  # fall back to full PAM

    # Precompute full pairwise distances for assignment scoring
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    full_pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    full_dist = pairwise_to_dict(full_pairwise)
    str_ids = [str(i) for i in ids]
    for sid in str_ids:
        full_dist[(sid, sid)] = 0.0

    rng = random.Random(seed)
    best_labels: pl.DataFrame | None = None
    best_cost = float("inf")

    for i in range(n_samples):
        # Subsample
        sample_ids = rng.sample(ids, effective_sample) if effective_sample < n else list(ids)
        sub_df = df.filter(pl.col(id_col).is_in(sample_ids))

        # Run PAM on subsample
        sub_labels = kmedoids(
            sub_df,
            k=k,
            method=method,
            max_iter=max_iter,
            seed=seed + i,
            id_col=id_col,
            target_col=target_col,
            **distance_kwargs,
        )

        # Extract medoids from subsample result
        medoid_map: dict[int, list[str]] = {}
        for row in sub_labels.to_dicts():
            cid = row["cluster"]
            uid = str(row[id_col])
            medoid_map.setdefault(cid, []).append(uid)

        medoids: list[str] = []
        for cid in sorted(medoid_map):
            members = medoid_map[cid]
            best_m = min(
                members,
                key=lambda m: sum(full_dist.get((m, o), 0.0) for o in members),
            )
            medoids.append(best_m)

        # Assign ALL series to nearest medoid
        assignments = []
        for sid in str_ids:
            best_cluster = min(
                range(len(medoids)),
                key=lambda ci: full_dist.get((sid, medoids[ci]), float("inf")),
            )
            assignments.append(best_cluster)

        # Evaluate full-dataset cost
        cost = sum(full_dist.get((sid, medoids[assignments[j]]), 0.0) for j, sid in enumerate(str_ids))

        if cost < best_cost:
            best_cost = cost
            best_labels = pl.DataFrame(
                {id_col: ids, "cluster": assignments},
                schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
            )

    assert best_labels is not None
    return best_labels


def clarans(
    df: pl.DataFrame,
    k: int,
    method: str = "dtw",
    num_local: int = 2,
    max_neighbor: int = 10,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """CLARANS: randomized medoid neighborhood search.

    Performs ``num_local`` restarts of a local search that explores up to
    ``max_neighbor`` random medoid swaps before declaring convergence.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of clusters.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``). Default ``"dtw"``.
    num_local
        Number of random restarts. Default 2.
    max_neighbor
        Maximum random swap attempts per restart before stopping. Default 10.
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
    ids = df[id_col].unique().sort().to_list()
    n = len(ids)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of series ({n})")

    # Precompute full pairwise distances
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    full_pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist = pairwise_to_dict(full_pairwise)
    str_ids = [str(i) for i in ids]
    for sid in str_ids:
        dist[(sid, sid)] = 0.0

    def _assign(medoids: list[str]) -> list[int]:
        return [min(range(len(medoids)), key=lambda ci: dist.get((sid, medoids[ci]), float("inf"))) for sid in str_ids]

    def _cost(medoids: list[str], assignments: list[int]) -> float:
        return sum(dist.get((sid, medoids[assignments[j]]), 0.0) for j, sid in enumerate(str_ids))

    rng = random.Random(seed)
    best_medoids: list[str] = []
    best_assignments: list[int] = []
    best_cost = float("inf")

    for local_i in range(num_local):
        # Random initial medoids
        local_rng = random.Random(seed + local_i)
        current_medoids = local_rng.sample(str_ids, k)
        current_assignments = _assign(current_medoids)
        current_cost = _cost(current_medoids, current_assignments)

        # Randomized neighborhood search
        neighbor_count = 0
        while neighbor_count < max_neighbor:
            # Pick a random medoid to swap
            swap_idx = rng.randint(0, k - 1)
            # Pick a random non-medoid to swap in
            non_medoids = [s for s in str_ids if s not in current_medoids]
            if not non_medoids:
                break
            candidate = rng.choice(non_medoids)

            new_medoids = list(current_medoids)
            new_medoids[swap_idx] = candidate
            new_assignments = _assign(new_medoids)
            new_cost = _cost(new_medoids, new_assignments)

            if new_cost < current_cost - 1e-12:
                current_medoids = new_medoids
                current_assignments = new_assignments
                current_cost = new_cost
                neighbor_count = 0  # reset on improvement
            else:
                neighbor_count += 1

        if current_cost < best_cost:
            best_cost = current_cost
            best_medoids = current_medoids
            best_assignments = current_assignments

    # Re-label clusters so labels are contiguous 0..k-1 sorted by medoid name
    sorted_medoids = sorted(best_medoids)
    old_to_new = {i: sorted_medoids.index(best_medoids[i]) for i in range(k)}
    final_assignments = [old_to_new[a] for a in best_assignments]

    return pl.DataFrame(
        {id_col: ids, "cluster": final_assignments},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )
