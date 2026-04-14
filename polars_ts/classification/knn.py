"""k-Nearest Neighbors classifier for time series using any distance metric."""

from __future__ import annotations

from collections import Counter
from typing import Any

import polars as pl

from polars_ts.distance import compute_pairwise_distance


class TimeSeriesKNNClassifier:
    """k-Nearest Neighbors classifier for time series.

    Uses the unified distance API to compute pairwise distances between
    time series, then classifies based on majority vote of k nearest neighbors.

    Args:
        k: Number of nearest neighbors. Default 1.
        metric: Distance metric name (any metric supported by
            ``compute_pairwise_distance``). Default ``"dtw"``.
        **metric_kwargs: Additional keyword arguments passed to the distance function.

    Examples:
        >>> clf = TimeSeriesKNNClassifier(k=1, metric="dtw")
        >>> clf.fit(train_df, label_col="label")
        >>> predictions = clf.predict(test_df)

    """

    def __init__(self, k: int = 1, metric: str = "dtw", **metric_kwargs: Any) -> None:
        self.k = k
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self._train_df: pl.DataFrame | None = None
        self._labels: dict[str, str] = {}

    def fit(self, df: pl.DataFrame, *, label_col: str = "label") -> TimeSeriesKNNClassifier:
        """Fit the classifier with labeled training data.

        Args:
            df: DataFrame with columns ``unique_id``, ``y``, and a label column.
                Each ``unique_id`` should have exactly one label.
            label_col: Name of the column containing class labels.

        Returns:
            self

        """
        labels_df = df.select("unique_id", label_col).unique(subset=["unique_id"])
        self._labels = dict(
            zip(
                labels_df["unique_id"].cast(pl.String).to_list(),
                labels_df[label_col].cast(pl.String).to_list(),
            )
        )
        self._train_df = df.select("unique_id", "y")
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict class labels for test time series.

        Args:
            df: DataFrame with columns ``unique_id`` and ``y``.

        Returns:
            DataFrame with columns ``unique_id`` and ``predicted_label``.

        """
        if self._train_df is None:
            raise RuntimeError("Call fit() before predict()")

        test_df = df.select("unique_id", "y")
        distances = compute_pairwise_distance(
            test_df, self._train_df, method=self.metric, **self.metric_kwargs
        )

        dist_col = [c for c in distances.columns if c not in ("id_1", "id_2")][0]
        predictions: dict[str, str] = {}

        test_ids = test_df["unique_id"].unique().cast(pl.String).to_list()
        for test_id in test_ids:
            # Get distances from this test series to all training series
            dists = distances.filter(
                (pl.col("id_1").cast(pl.String) == test_id)
                | (pl.col("id_2").cast(pl.String) == test_id)
            )

            # Build (train_id, distance) pairs
            neighbors: list[tuple[str, float]] = []
            for row in dists.to_dicts():
                id1 = str(row["id_1"])
                id2 = str(row["id_2"])
                train_id = id2 if id1 == test_id else id1
                if train_id in self._labels:
                    neighbors.append((train_id, row[dist_col]))

            # Sort by distance, take k nearest
            neighbors.sort(key=lambda x: x[1])
            k_nearest = neighbors[: self.k]

            # Majority vote
            if k_nearest:
                votes = Counter(self._labels[n[0]] for n in k_nearest)
                predictions[test_id] = votes.most_common(1)[0][0]

        pred_ids = list(predictions.keys())
        pred_labels = [predictions[uid] for uid in pred_ids]
        return pl.DataFrame({"unique_id": pred_ids, "predicted_label": pred_labels})
