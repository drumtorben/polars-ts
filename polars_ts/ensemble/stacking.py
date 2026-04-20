"""Stacking forecaster using a meta-learner.

Trains a meta-learner on out-of-fold predictions from multiple base
models, then combines new forecasts through the fitted meta-learner.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_ts.ensemble.weighted import _join_forecasts
from polars_ts.models.multistep import Estimator


class StackingForecaster:
    """Stacking ensemble that trains a meta-learner on base model predictions.

    Parameters
    ----------
    meta_learner
        A scikit-learn-compatible estimator with ``fit`` and ``predict``.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with actual target values.

    """

    def __init__(
        self,
        meta_learner: Estimator,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.meta_learner = meta_learner
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.is_fitted_: bool = False
        self.n_models_: int = 0

    def fit(
        self,
        cv_predictions: list[pl.DataFrame],
        actuals: pl.DataFrame,
    ) -> StackingForecaster:
        """Train the meta-learner on out-of-fold base model predictions.

        Parameters
        ----------
        cv_predictions
            List of DataFrames, one per base model, each containing
            out-of-fold predictions with columns ``[id_col, time_col, "y_hat"]``.
        actuals
            DataFrame with actual values in ``target_col``, keyed by
            ``id_col`` and ``time_col``.

        Returns
        -------
        StackingForecaster
            Fitted forecaster (``self``).

        """
        if not cv_predictions:
            raise ValueError("cv_predictions must be a non-empty list")
        if len(cv_predictions) < 2:
            raise ValueError("Need at least 2 base models for stacking")

        self.n_models_ = len(cv_predictions)

        # Join all CV predictions into feature matrix
        joined = _join_forecasts(cv_predictions, self.id_col, self.time_col)

        # Join with actuals to get target
        join_cols = [c for c in [self.id_col, self.time_col] if c in joined.columns]
        with_target = joined.join(
            actuals.select(*join_cols, self.target_col),
            on=join_cols,
            how="inner",
        )

        if len(with_target) == 0:
            raise ValueError("No matching rows between cv_predictions and actuals")

        # Build numpy arrays
        hat_cols = [f"y_hat_{i}" for i in range(self.n_models_)]
        X = with_target.select(hat_cols).to_numpy().astype(np.float64)
        y = with_target[self.target_col].to_numpy().astype(np.float64)

        self.meta_learner.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, forecasts: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine base model forecasts through the fitted meta-learner.

        Parameters
        ----------
        forecasts
            List of forecast DataFrames, one per base model, each with
            ``[id_col, time_col, "y_hat"]``.

        Returns
        -------
        pl.DataFrame
            Combined forecast with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if len(forecasts) != self.n_models_:
            raise ValueError(f"Expected {self.n_models_} forecasts, got {len(forecasts)}")

        joined = _join_forecasts(forecasts, self.id_col, self.time_col)
        hat_cols = [f"y_hat_{i}" for i in range(self.n_models_)]
        X = joined.select(hat_cols).to_numpy().astype(np.float64)

        preds = self.meta_learner.predict(X)

        join_cols = [c for c in [self.id_col, self.time_col] if c in joined.columns]
        result = joined.select(join_cols).with_columns(
            pl.Series("y_hat", preds.astype(np.float64)),
        )
        return result
