"""K-Nearest Neighbors classification for time series using precomputed distances."""

from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict


def knn_classify(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    k: int = 3,
    method: str = "dtw",
    id_col: str = "unique_id",
    target_col: str = "y",
    label_col: str = "label",
    **distance_kwargs: Any,
) -> pl.DataFrame:
    """K-Nearest Neighbors time series classification.

    Parameters
    ----------
    train_df
        Training DataFrame with columns ``id_col``, ``target_col``, and ``label_col``.
        The ``label_col`` must have one label per ``id_col``.
    test_df
        Test DataFrame with columns ``id_col`` and ``target_col``.
    k
        Number of nearest neighbors.
    method
        Distance metric name (e.g. ``"dtw"``, ``"erp"``, ``"lcss"``).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.
    label_col
        Column with class labels in the training data.
    **distance_kwargs
        Extra keyword arguments forwarded to the distance function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "predicted_label"]``.

    """
    if label_col not in train_df.columns:
        raise ValueError(f"Training data must contain label column {label_col!r}")

    # Build label lookup: str(id) -> label
    label_df = (
        train_df.group_by(id_col)
        .agg(pl.col(label_col).first())
        .select(pl.col(id_col).cast(pl.String).alias("_id"), label_col)
    )
    label_map = dict(zip(label_df["_id"].to_list(), label_df[label_col].to_list(), strict=False))

    # Prepare DataFrames for distance computation (just id + values)
    train_dist = train_df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))
    test_dist = test_df.select(pl.col(id_col).alias("unique_id"), pl.col(target_col).alias("y"))

    # Compute cross-distances
    pairwise = compute_distances(train_dist, test_dist, method=method, **distance_kwargs)
    dist = pairwise_to_dict(pairwise)

    train_ids = set(train_dist["unique_id"].unique().to_list())
    test_ids = test_df[id_col].unique().sort().to_list()
    test_str_ids = [str(tid) for tid in test_ids]

    # Classify each test series
    predictions = []
    for test_id, test_str in zip(test_ids, test_str_ids, strict=False):
        # Get distances to all training series
        distances = []
        for train_id in train_ids:
            d = dist.get((test_str, str(train_id)))
            if d is not None:
                distances.append((train_id, d))

        # Sort by distance, take k nearest
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]

        # Majority vote
        votes = Counter(label_map[str(n[0])] for n in neighbors)
        predicted = votes.most_common(1)[0][0]
        predictions.append((test_id, predicted))

    return pl.DataFrame(
        {
            id_col: [p[0] for p in predictions],
            "predicted_label": [p[1] for p in predictions],
        },
        schema={id_col: test_df[id_col].dtype, "predicted_label": train_df[label_col].dtype},
    )
