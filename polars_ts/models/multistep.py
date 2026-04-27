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
    past_covariates
        Column names of time-varying covariates known only up to the
        present.  Lagged features are created for each covariate.
    future_covariates
        Column names of time-varying covariates known into the future.
        Values for the forecast horizon must be supplied in *future_df*
        at predict time.
    past_covariate_lags
        Lag offsets for past covariate features.  Defaults to *lags*.
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
        past_covariates: list[str] | None = None,
        future_covariates: list[str] | None = None,
        past_covariate_lags: list[int] | None = None,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not lags or any(k <= 0 for k in lags):
            raise ValueError("lags must be a non-empty list of positive integers")
        self.estimator = estimator
        self.lags = sorted(lags)
        self.past_covariates = sorted(past_covariates) if past_covariates else None
        self.future_covariates = sorted(future_covariates) if future_covariates else None
        self.past_covariate_lags = sorted(past_covariate_lags) if past_covariate_lags else None
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
            Training DataFrame with at least ``id_col``, ``time_col``,
            ``target_col``, and any covariate columns.

        Returns
        -------
        RecursiveForecaster
            Fitted forecaster (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        cov_lags = self.past_covariate_lags or self.lags
        max_lag = max(max(self.lags), max(cov_lags) if cov_lags else 0)
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for _group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            n = len(values)
            if n <= max_lag:
                continue

            # Pre-extract covariate lists to avoid repeated .to_list() calls
            pcov_lists = {col: group_df[col].to_list() for col in self.past_covariates} if self.past_covariates else {}
            fcov_lists = (
                {col: group_df[col].to_list() for col in self.future_covariates} if self.future_covariates else {}
            )

            rows_x: list[list[float]] = []
            rows_y: list[float] = []
            for t in range(max_lag, n):
                row: list[float] = [values[t - lag] for lag in self.lags]
                for col in self.past_covariates or []:
                    for lag in cov_lags:
                        row.append(pcov_lists[col][t - lag])
                for col in self.future_covariates or []:
                    row.append(float(fcov_lists[col][t]))
                rows_x.append(row)
                rows_y.append(values[t])
            if rows_x:
                all_x.append(np.array(rows_x, dtype=np.float64))
                all_y.append(np.array(rows_y, dtype=np.float64))

        if not all_x:
            raise ValueError("No training samples — series are shorter than max(lags)")

        X = np.vstack(all_x)
        Y = np.concatenate(all_y)
        self.estimator.fit(X, Y)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        df: pl.DataFrame,
        h: int,
        future_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Generate *h*-step-ahead forecasts by recursive prediction.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.
        h
            Forecast horizon (number of steps ahead).
        future_df
            DataFrame with future covariate values for the forecast
            horizon.  Required when *future_covariates* were specified
            at construction.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("Horizon h must be a positive integer")
        if self.future_covariates and future_df is None:
            raise ValueError("future_df is required when future_covariates are configured")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])
        cov_lags = self.past_covariate_lags or self.lags
        max_lag = max(max(self.lags), max(cov_lags) if cov_lags else 0)

        # Build future covariate lookup
        future_lookup: dict[tuple[Any, Any], dict[str, float]] = {}
        if self.future_covariates and future_df is not None:
            for row in future_df.iter_rows(named=True):
                key = (row[self.id_col], row[self.time_col])
                future_lookup[key] = {col: float(row[col]) for col in self.future_covariates}

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = group_df[self.target_col].to_list()
            last_time = group_df[self.time_col][-1]
            buffer = list(values[-max_lag:])
            future_times = _make_future_dates(last_time, freq, h)

            # Past covariate buffers
            pcov_buffers: dict[str, list[float]] | None = None
            if self.past_covariates:
                pcov_buffers = {col: group_df[col].to_list()[-max_lag:] for col in self.past_covariates}

            for step in range(h):
                row_feats: list[float] = [buffer[-lag] for lag in self.lags]
                if self.past_covariates and pcov_buffers:
                    for col in self.past_covariates:
                        buf = pcov_buffers[col]
                        for lag in cov_lags:
                            if lag <= len(buf):
                                row_feats.append(buf[-lag])
                            else:
                                row_feats.append(buf[0] if buf else 0.0)
                if self.future_covariates:
                    key = (gid, future_times[step])
                    fcov = future_lookup.get(key, {col: 0.0 for col in self.future_covariates})
                    for col in self.future_covariates:
                        row_feats.append(fcov[col])

                x_row = np.array([row_feats], dtype=np.float64)
                pred = float(self.estimator.predict(x_row)[0])
                buffer.append(pred)
                rows.append({self.id_col: gid, self.time_col: future_times[step], "y_hat": pred})

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
    past_covariates
        Column names of time-varying covariates known only up to the
        present.  Lagged features are created for each covariate.
    future_covariates
        Column names of time-varying covariates known into the future.
        Values for the forecast horizon must be supplied in *future_df*
        at predict time.
    past_covariate_lags
        Lag offsets for past covariate features.  Defaults to *lags*.
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
        past_covariates: list[str] | None = None,
        future_covariates: list[str] | None = None,
        past_covariate_lags: list[int] | None = None,
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
        self.past_covariates = sorted(past_covariates) if past_covariates else None
        self.future_covariates = sorted(future_covariates) if future_covariates else None
        self.past_covariate_lags = sorted(past_covariate_lags) if past_covariate_lags else None
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.estimators_: list[Estimator] = []

    def fit(self, df: pl.DataFrame) -> DirectForecaster:
        """Fit *h* models, where model *k* predicts the value *k* steps ahead.

        Parameters
        ----------
        df
            Training DataFrame with at least ``id_col``, ``time_col``,
            ``target_col``, and any covariate columns.

        Returns
        -------
        DirectForecaster
            Fitted forecaster (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        cov_lags = self.past_covariate_lags or self.lags
        max_lag = max(max(self.lags), max(cov_lags) if cov_lags else 0)

        self.estimators_ = []
        for horizon_k in range(1, self.h + 1):
            all_x: list[np.ndarray] = []
            all_y: list[np.ndarray] = []

            for _group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
                values = group_df[self.target_col].to_list()
                n = len(values)
                if n <= max_lag + horizon_k:
                    continue
                # Pre-extract covariate lists
                pcov_lists = (
                    {col: group_df[col].to_list() for col in self.past_covariates} if self.past_covariates else {}
                )
                fcov_lists = (
                    {col: group_df[col].to_list() for col in self.future_covariates} if self.future_covariates else {}
                )

                rows_x: list[list[float]] = []
                rows_y: list[float] = []
                for t in range(max_lag, n - horizon_k):
                    row: list[float] = [values[t - lag] for lag in self.lags]
                    for col in self.past_covariates or []:
                        for lag in cov_lags:
                            row.append(pcov_lists[col][t - lag])
                    for col in self.future_covariates or []:
                        row.append(float(fcov_lists[col][t + horizon_k]))
                    rows_x.append(row)
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

    def predict(
        self,
        df: pl.DataFrame,
        future_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Generate forecasts for horizons 1 through *h*.

        Each fitted model predicts its horizon from the last available
        observation's lag features. No recursive feeding is needed.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.
        future_df
            DataFrame with future covariate values for the forecast
            horizon.  Required when *future_covariates* were specified
            at construction.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict()")
        if self.future_covariates and future_df is None:
            raise ValueError("future_df is required when future_covariates are configured")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])
        cov_lags = self.past_covariate_lags or self.lags

        # Build future covariate lookup
        future_lookup: dict[tuple[Any, Any], dict[str, float]] = {}
        if self.future_covariates and future_df is not None:
            for row in future_df.iter_rows(named=True):
                key = (row[self.id_col], row[self.time_col])
                future_lookup[key] = {col: float(row[col]) for col in self.future_covariates}

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = group_df[self.target_col].to_list()
            last_time = group_df[self.time_col][-1]
            future_times = _make_future_dates(last_time, freq, self.h)

            # Build feature row from the last observation (reference point = n-1)
            x_base: list[float] = [values[-(1 + lag)] for lag in self.lags]
            if self.past_covariates:
                for col in self.past_covariates:
                    cov_vals = group_df[col].to_list()
                    for lag in cov_lags:
                        idx = len(cov_vals) - 1 - lag
                        x_base.append(cov_vals[idx] if idx >= 0 else cov_vals[0])

            for step, estimator in enumerate(self.estimators_):
                row_feats = list(x_base)
                if self.future_covariates:
                    key = (gid, future_times[step])
                    fcov = future_lookup.get(key, {col: 0.0 for col in self.future_covariates})
                    for col in self.future_covariates:
                        row_feats.append(fcov[col])

                x_row = np.array([row_feats], dtype=np.float64)
                pred = float(estimator.predict(x_row)[0])
                rows.append({self.id_col: gid, self.time_col: future_times[step], "y_hat": pred})

        schema = {self.id_col: df.schema[self.id_col], self.time_col: df.schema[self.time_col], "y_hat": pl.Float64()}
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)
