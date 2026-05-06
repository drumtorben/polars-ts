"""ROCKET and MiniROCKET classifiers for time series.

Uses existing ROCKET/MiniROCKET feature extraction + Ridge classifier
for fast and accurate time series classification.

References
----------
Dempster et al. (2020). *ROCKET: Exceptionally fast and accurate
time series classification using random convolutional kernels.* DMKD.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.features.rocket import minirocket_features, rocket_features


class RocketClassifier:
    """Time series classifier using ROCKET features + Ridge.

    Extracts ROCKET features from each series, then trains a Ridge
    classifier on the feature vectors.

    Parameters
    ----------
    n_kernels
        Number of random convolutional kernels.
    alpha
        Ridge regularization strength.
    seed
        Random seed for kernel generation.
    id_col, target_col, time_col
        Column names.

    """

    def __init__(
        self,
        n_kernels: int = 500,
        alpha: float = 1.0,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        self.n_kernels = n_kernels
        self.alpha = alpha
        self.seed = seed
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self._classifier: Any = None
        self._label_encoder: dict[str, int] = {}
        self._label_decoder: dict[int, str] = {}
        self._label_col: str = "label"
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame, *, label_col: str = "label") -> RocketClassifier:
        """Fit the classifier on training data.

        Parameters
        ----------
        df
            Training DataFrame with ``id_col``, ``target_col``, and ``label_col``.
        label_col
            Column containing class labels.

        Returns
        -------
        RocketClassifier
            Self, for chaining.

        """
        from sklearn.linear_model import RidgeClassifier

        self._label_col = label_col

        # Extract features
        feat_df = rocket_features(
            df,
            n_kernels=self.n_kernels,
            target_col=self.target_col,
            id_col=self.id_col,
            time_col=self.time_col,
            seed=self.seed,
        )

        # Get labels per series
        labels_df = df.group_by(self.id_col).agg(pl.col(label_col).first())
        feat_with_labels = feat_df.join(labels_df, on=self.id_col)

        # Encode labels
        unique_labels = sorted(feat_with_labels[label_col].unique().to_list())
        self._label_encoder = {lbl: i for i, lbl in enumerate(unique_labels)}
        self._label_decoder = {i: lbl for lbl, i in self._label_encoder.items()}

        # Build feature matrix and label vector
        feat_cols = [c for c in feat_df.columns if c != self.id_col]
        X = feat_with_labels.select(feat_cols).to_numpy()
        y = np.array([self._label_encoder[lbl] for lbl in feat_with_labels[label_col].to_list()])

        # Train Ridge classifier
        self._classifier = RidgeClassifier(alpha=self.alpha)
        self._classifier.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict class labels for test time series.

        Parameters
        ----------
        df
            Test DataFrame with ``id_col`` and ``target_col``.

        Returns
        -------
        pl.DataFrame
            Columns: ``unique_id``, ``predicted_label``.

        """
        if not self.is_fitted_ or self._classifier is None:
            raise RuntimeError("Call fit() before predict().")

        feat_df = rocket_features(
            df,
            n_kernels=self.n_kernels,
            target_col=self.target_col,
            id_col=self.id_col,
            time_col=self.time_col,
            seed=self.seed,
        )

        ids = feat_df[self.id_col].to_list()
        feat_cols = [c for c in feat_df.columns if c != self.id_col]
        X = feat_df.select(feat_cols).to_numpy()

        y_pred = self._classifier.predict(X)
        labels = [self._label_decoder[int(yi)] for yi in y_pred]

        return pl.DataFrame({self.id_col: ids, "predicted_label": labels})


class MiniRocketClassifier:
    """Time series classifier using MiniROCKET features + Ridge.

    Same pattern as :class:`RocketClassifier` but uses the faster
    MiniROCKET feature extraction.

    Parameters
    ----------
    n_kernels
        Number of kernels.
    alpha
        Ridge regularization strength.
    seed
        Random seed.
    id_col, target_col, time_col
        Column names.

    """

    def __init__(
        self,
        n_kernels: int = 500,
        alpha: float = 1.0,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        self.n_kernels = n_kernels
        self.alpha = alpha
        self.seed = seed
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self._classifier: Any = None
        self._label_encoder: dict[str, int] = {}
        self._label_decoder: dict[int, str] = {}
        self._label_col: str = "label"
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame, *, label_col: str = "label") -> MiniRocketClassifier:
        """Fit the classifier on training data."""
        from sklearn.linear_model import RidgeClassifier

        self._label_col = label_col

        feat_df = minirocket_features(
            df,
            n_kernels=self.n_kernels,
            target_col=self.target_col,
            id_col=self.id_col,
            time_col=self.time_col,
            seed=self.seed,
        )

        labels_df = df.group_by(self.id_col).agg(pl.col(label_col).first())
        feat_with_labels = feat_df.join(labels_df, on=self.id_col)

        unique_labels = sorted(feat_with_labels[label_col].unique().to_list())
        self._label_encoder = {lbl: i for i, lbl in enumerate(unique_labels)}
        self._label_decoder = {i: lbl for lbl, i in self._label_encoder.items()}

        feat_cols = [c for c in feat_df.columns if c != self.id_col]
        X = feat_with_labels.select(feat_cols).to_numpy()
        y = np.array([self._label_encoder[lbl] for lbl in feat_with_labels[label_col].to_list()])

        self._classifier = RidgeClassifier(alpha=self.alpha)
        self._classifier.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict class labels for test time series."""
        if not self.is_fitted_ or self._classifier is None:
            raise RuntimeError("Call fit() before predict().")

        feat_df = minirocket_features(
            df,
            n_kernels=self.n_kernels,
            target_col=self.target_col,
            id_col=self.id_col,
            time_col=self.time_col,
            seed=self.seed,
        )

        ids = feat_df[self.id_col].to_list()
        feat_cols = [c for c in feat_df.columns if c != self.id_col]
        X = feat_df.select(feat_cols).to_numpy()

        y_pred = self._classifier.predict(X)
        labels = [self._label_decoder[int(yi)] for yi in y_pred]

        return pl.DataFrame({self.id_col: ids, "predicted_label": labels})
