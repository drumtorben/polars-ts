"""k-Shape nearest-centroid classifier for time series."""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_ts.clustering.kshape import KShape, _sbd, _zscore


class KShapeClassifier:
    """Nearest-centroid classifier using SBD and k-Shape centroids.

    Fits k-Shape clustering per class on training data to extract
    representative centroids, then classifies test series by nearest
    centroid distance (SBD).

    Args:
        n_centroids_per_class: Number of k-Shape centroids per class. Default 1.
        max_iter: Maximum k-Shape iterations per class. Default 100.

    Examples:
        >>> clf = KShapeClassifier(n_centroids_per_class=1)
        >>> clf.fit(train_df, label_col="label")
        >>> predictions = clf.predict(test_df)

    """

    def __init__(self, n_centroids_per_class: int = 1, max_iter: int = 100) -> None:
        self.n_centroids_per_class = n_centroids_per_class
        self.max_iter = max_iter
        self._centroids: list[tuple[str, np.ndarray]] = []

    def fit(
        self, df: pl.DataFrame, *, label_col: str = "label"
    ) -> KShapeClassifier:
        """Fit the classifier by computing k-Shape centroids for each class.

        Args:
            df: DataFrame with columns ``unique_id``, ``y``, and a label column.
            label_col: Name of the column containing class labels.

        Returns:
            self

        """
        labels_df = df.select("unique_id", label_col).unique(subset=["unique_id"])
        label_map = dict(
            zip(
                labels_df["unique_id"].cast(pl.String).to_list(),
                labels_df[label_col].cast(pl.String).to_list(),
            )
        )

        classes = sorted(set(label_map.values()))
        self._centroids = []

        for cls in classes:
            cls_ids = [uid for uid, lbl in label_map.items() if lbl == cls]
            cls_df = df.filter(pl.col("unique_id").cast(pl.String).is_in(cls_ids)).select(
                "unique_id", "y"
            )

            n_series = len(cls_ids)
            n_centroids = min(self.n_centroids_per_class, n_series)

            if n_centroids == 1:
                # Just compute mean z-normalized shape
                series_list = []
                for uid in cls_ids:
                    vals = cls_df.filter(
                        pl.col("unique_id").cast(pl.String) == uid
                    )["y"].to_numpy()
                    series_list.append(_zscore(vals.astype(np.float64)))
                centroid = np.mean(series_list, axis=0)
                self._centroids.append((cls, _zscore(centroid)))
            else:
                ks = KShape(n_clusters=n_centroids, max_iter=self.max_iter)
                ks.fit(cls_df)
                for c in ks.centroids_:
                    self._centroids.append((cls, c))

        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict class labels for test time series.

        Args:
            df: DataFrame with columns ``unique_id`` and ``y``.

        Returns:
            DataFrame with columns ``unique_id`` and ``predicted_label``.

        """
        if not self._centroids:
            raise RuntimeError("Call fit() before predict()")

        predictions: dict[str, str] = {}
        for uid in sorted(df["unique_id"].unique().cast(pl.String).to_list()):
            vals = df.filter(pl.col("unique_id").cast(pl.String) == uid)["y"].to_numpy()
            series = _zscore(vals.astype(np.float64))

            best_label = self._centroids[0][0]
            best_dist = float("inf")
            for label, centroid in self._centroids:
                # Pad to same length if needed
                max_len = max(len(series), len(centroid))
                s = np.pad(series, (0, max_len - len(series))) if len(series) < max_len else series
                c = np.pad(centroid, (0, max_len - len(centroid))) if len(centroid) < max_len else centroid
                dist, _ = _sbd(s, c)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            predictions[uid] = best_label

        pred_ids = list(predictions.keys())
        pred_labels = [predictions[uid] for uid in pred_ids]
        return pl.DataFrame({"unique_id": pred_ids, "predicted_label": pred_labels})
