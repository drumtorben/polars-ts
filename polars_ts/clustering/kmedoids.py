"""K-Medoids (PAM) clustering for time series using precomputed distances."""

from __future__ import annotations

import random
from typing import Any

import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


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

    # Rename columns if needed to match the expected format
    dist_df = df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    pairwise = compute_distances(dist_df, dist_df, method=method, **distance_kwargs)
    dist = pairwise_to_dict(pairwise)

    # Self-distance is 0
    str_ids = [str(i) for i in ids]
    for sid in str_ids:
        dist[(sid, sid)] = 0.0

    def _total_cost(assignments: dict[str, str]) -> float:
        return sum(dist[(sid, assignments[sid])] for sid in str_ids)

    def _assign(medoids: list[str]) -> dict[str, str]:
        assignment: dict[str, str] = {}
        for sid in str_ids:
            best = min(medoids, key=lambda m: dist[(sid, m)])
            assignment[sid] = best
        return assignment

    # Initialize medoids randomly
    rng = random.Random(seed)
    medoids = rng.sample(str_ids, k)

    assignments = _assign(medoids)
    current_cost = _total_cost(assignments)

    # PAM swap loop
    for _ in range(max_iter):
        improved = False
        for i, _med in enumerate(medoids):
            non_medoids = [s for s in str_ids if s not in medoids]
            for candidate in non_medoids:
                new_medoids = medoids[:i] + [candidate] + medoids[i + 1 :]
                new_assignments = _assign(new_medoids)
                new_cost = _total_cost(new_assignments)
                if new_cost < current_cost - 1e-12:
                    medoids = new_medoids
                    assignments = new_assignments
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Map medoids to cluster labels 0..k-1
    medoid_to_label = {m: i for i, m in enumerate(sorted(medoids))}
    rows = [(orig_id, medoid_to_label[assignments[str(orig_id)]]) for orig_id in ids]

    return pl.DataFrame(
        {id_col: [r[0] for r in rows], "cluster": [r[1] for r in rows]},
        schema={id_col: df[id_col].dtype, "cluster": pl.Int64},
    )
