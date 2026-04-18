"""Multi-step forecasting strategies: recursive and direct.

Implements model-agnostic wrappers for multi-step-ahead forecasting
from Ch 18 of "Modern Time Series Forecasting with Python" (2nd Ed.).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates


@runtime_checkable
class Estimator(Protocol):
    """Minimal interface for a scikit-learn-compatible estimator."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _build_lag_matrix(
    values: list[float],
    lags: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and target vector from a value sequence.

    Parameters
    ----------
    values
        Ordered list of target values.
    lags
        Lag offsets (positive integers). For each row *t*, feature *i* is
        ``values[t - lags[i]]``.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(n_valid_rows, len(lags))``.
    y : np.ndarray
        Target vector of shape ``(n_valid_rows,)``.

    """
    max_lag = max(lags)
    n = len(values)
    if n <= max_lag:
        return np.empty((0, len(lags))), np.empty((0,))

    rows_x: list[list[float]] = []
    rows_y: list[float] = []
    for t in range(max_lag, n):
        rows_x.append([values[t - lag] for lag in lags])
        rows_y.append(values[t])
    return np.array(rows_x, dtype=np.float64), np.array(rows_y, dtype=np.float64)


class RecursiveForecaster:
    """Recursive multi-step forecaster.

    Train a single 1-step-ahead model. At prediction time, feed each
    prediction back as input to generate the next step, up to horizon *h*.

    Parameters
    ----------
    estimator
        A scikit-learn-compatible estimator with ``fit`` and ``predict``.
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
        estimator: Estimator,
        lags: list[int],
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not lags or any(k <= 0 for k in lags):
            raise ValueError("lags must be a non-empty list of positive integers")
        self.estimator = estimator
        self.lags = sorted(lags)
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> RecursiveForecaster:
        """Fit the estimator on lag features derived from *df*.

        All series are pooled together for a single global model.

        Parameters
        ----------
        df
            Training DataFrame with at least ``id_col``, ``time_col``, and
            ``target_col``.

        Returns
        -------
        RecursiveForecaster
            Fitted forecaster (``self``).

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
        self.estimator.fit(X, Y)
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate *h*-step-ahead forecasts by recursive prediction.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.
        h
            Forecast horizon (number of steps ahead).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("Horizon h must be a positive integer")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])
        max_lag = max(self.lags)

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            last_time = group_df[self.time_col][-1]
            # Buffer: keep at least max_lag values
            buffer = list(values[-max_lag:])
            future_times = _make_future_dates(last_time, freq, h)

            for step in range(h):
                x_row = np.array([[buffer[-lag] for lag in self.lags]], dtype=np.float64)
                pred = float(self.estimator.predict(x_row)[0])
                buffer.append(pred)
                rows.append({self.id_col: group_id[0], self.time_col: future_times[step], "y_hat": pred})

        schema = {self.id_col: df.schema[self.id_col], self.time_col: df.schema[self.time_col], "y_hat": pl.Float64()}
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)


class DirectForecaster:
    """Direct multi-step forecaster.

    Train *h* separate models, one per forecast horizon step.

    Parameters
    ----------
    estimator_factory
        Callable that returns a fresh estimator instance. Called *h* times.
        Example: ``lambda: LinearRegression()``.
    lags
        Lag offsets used as features.
    h
        Forecast horizon. Determines how many models are trained.
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    """

    def __init__(
        self,
        estimator_factory: Callable[[], Estimator],
        lags: list[int],
        h: int = 1,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not lags or any(k <= 0 for k in lags):
            raise ValueError("lags must be a non-empty list of positive integers")
        if h <= 0:
            raise ValueError("Horizon h must be a positive integer")
        self.estimator_factory = estimator_factory
        self.lags = sorted(lags)
        self.h = h
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.estimators_: list[Estimator] = []

    def fit(self, df: pl.DataFrame) -> DirectForecaster:
        """Fit *h* models, where model *k* predicts the value *k* steps ahead.

        Parameters
        ----------
        df
            Training DataFrame.

        Returns
        -------
        DirectForecaster
            Fitted forecaster (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        max_lag = max(self.lags)

        self.estimators_ = []
        for horizon_k in range(1, self.h + 1):
            all_x: list[np.ndarray] = []
            all_y: list[np.ndarray] = []

            for _group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
                values = group_df[self.target_col].to_list()
                n = len(values)
                # For horizon k: X row at time t uses lags from t, target is values[t + k]
                # t ranges from max_lag to n - k
                if n <= max_lag + horizon_k:
                    continue
                rows_x: list[list[float]] = []
                rows_y: list[float] = []
                for t in range(max_lag, n - horizon_k):
                    rows_x.append([values[t - lag] for lag in self.lags])
                    rows_y.append(values[t + horizon_k])
                if rows_x:
                    all_x.append(np.array(rows_x, dtype=np.float64))
                    all_y.append(np.array(rows_y, dtype=np.float64))

            if not all_x:
                raise ValueError(
                    f"No training samples for horizon {horizon_k} — "
                    "series are too short for the given lags and horizon"
                )

            X = np.vstack(all_x)
            Y = np.concatenate(all_y)
            estimator = self.estimator_factory()
            estimator.fit(X, Y)
            self.estimators_.append(estimator)

        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate forecasts for horizons 1 through *h*.

        Each fitted model predicts its horizon from the last available
        observation's lag features. No recursive feeding is needed.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict()")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            last_time = group_df[self.time_col][-1]
            future_times = _make_future_dates(last_time, freq, self.h)

            # Build feature row from the last observation (reference point = n-1)
            # In training, features at t are values[t - lag], so at t = n-1:
            x_row = np.array([[values[-(1 + lag)] for lag in self.lags]], dtype=np.float64)

            for step, estimator in enumerate(self.estimators_):
                pred = float(estimator.predict(x_row)[0])
                rows.append({self.id_col: group_id[0], self.time_col: future_times[step], "y_hat": pred})

        schema = {self.id_col: df.schema[self.id_col], self.time_col: df.schema[self.time_col], "y_hat": pl.Float64()}
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)
