"""Quantile regression forecaster for prediction intervals.

Trains one model per quantile level using any scikit-learn-compatible
estimator that supports quantile loss (e.g. ``GradientBoostingRegressor``
with ``loss='quantile'``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.models.multistep import Estimator, _build_lag_matrix


class QuantileRegressor:
    """Quantile regression forecaster producing prediction intervals.

    Trains one model per quantile level. At prediction time, generates
    recursive multi-step forecasts for each quantile, using the median
    (q=0.5) prediction as the recursive input.

    Parameters
    ----------
    estimator_factory
        Callable that takes a quantile level (float in (0, 1)) and
        returns a fresh scikit-learn-compatible estimator.
        Example: ``lambda q: GradientBoostingRegressor(loss='quantile', alpha=q)``.
    quantiles
        Quantile levels to predict (e.g. ``[0.1, 0.5, 0.9]``).
    lags
        Lag offsets used as features (e.g. ``[1, 2, 7]``).
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    """

    def __init__(
        self,
        estimator_factory: Callable[[float], Estimator],
        quantiles: list[float],
        lags: list[int],
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not quantiles:
            raise ValueError("quantiles must be a non-empty list")
        if any(q <= 0 or q >= 1 for q in quantiles):
            raise ValueError("All quantile levels must be in (0, 1)")
        if not lags or any(k <= 0 for k in lags):
            raise ValueError("lags must be a non-empty list of positive integers")
        self.estimator_factory = estimator_factory
        self.quantiles = sorted(quantiles)
        self.lags = sorted(lags)
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.estimators_: dict[float, Estimator] = {}

    def fit(self, df: pl.DataFrame) -> QuantileRegressor:
        """Fit one estimator per quantile on lag features derived from *df*.

        All series are pooled together for a single global model per quantile.

        Parameters
        ----------
        df
            Training DataFrame with at least ``id_col``, ``time_col``, and
            ``target_col``.

        Returns
        -------
        QuantileRegressor
            Fitted regressor (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for _group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            x, y = _build_lag_matrix(values, self.lags)
            if len(x) > 0:
                all_x.append(x)
                all_y.append(y)

        if not all_x:
            raise ValueError("No training samples — series are shorter than max(lags)")

        X = np.vstack(all_x)
        Y = np.concatenate(all_y)

        self.estimators_ = {}
        for q in self.quantiles:
            est = self.estimator_factory(q)
            est.fit(X, Y)
            self.estimators_[q] = est

        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate *h*-step-ahead quantile forecasts by recursive prediction.

        The median quantile (closest to 0.5) is used as the recursive
        input for subsequent steps.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.
        h
            Forecast horizon (number of steps ahead).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat", "q_0.1", ...]``.
            The ``y_hat`` column uses the median quantile prediction.
            Quantile columns are named ``q_<level>`` for CRPS compatibility.

        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("Horizon h must be a positive integer")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])
        max_lag = max(self.lags)

        # Find the quantile closest to 0.5 for recursive feeding
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            if len(values) < max_lag:
                raise ValueError(
                    f"Series {group_id[0]!r} has {len(values)} observations "
                    f"but max(lags)={max_lag} — too short to predict"
                )
            last_time = group_df[self.time_col][-1]
            buffer = list(values[-max_lag:])
            future_times = _make_future_dates(last_time, freq, h)

            for step in range(h):
                x_row = np.array([[buffer[-lag] for lag in self.lags]], dtype=np.float64)
                row: dict[str, Any] = {
                    self.id_col: group_id[0],
                    self.time_col: future_times[step],
                }
                for q in self.quantiles:
                    pred = float(self.estimators_[q].predict(x_row)[0])
                    row[f"q_{q}"] = pred
                # y_hat = median quantile prediction
                row["y_hat"] = row[f"q_{median_q}"]
                rows.append(row)
                # Feed median prediction back into the buffer
                buffer.append(row["y_hat"])

        schema: dict[str, Any] = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
        }
        for q in self.quantiles:
            schema[f"q_{q}"] = pl.Float64()
        schema["y_hat"] = pl.Float64()
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)
