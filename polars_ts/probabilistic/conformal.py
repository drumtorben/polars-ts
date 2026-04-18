"""Conformal prediction intervals for time series forecasts.

Provides distribution-free prediction intervals with finite-sample
coverage guarantees. Implements split conformal and EnbPI (Ensemble
Batch Prediction Intervals) from Ch 16 of
"Modern Time Series Forecasting with Python" (2nd Ed.).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.models.multistep import Estimator, _build_lag_matrix


def conformal_interval(
    cal_residuals: pl.DataFrame,
    predictions: pl.DataFrame,
    coverage: float = 0.9,
    residual_col: str = "residual",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
    symmetric: bool = True,
) -> pl.DataFrame:
    """Add conformal prediction intervals to point forecasts.

    Uses calibration residuals to compute a coverage-adjusted quantile,
    then adds lower/upper bounds to the prediction DataFrame.

    Parameters
    ----------
    cal_residuals
        DataFrame containing calibration residuals. For symmetric mode,
        this should contain absolute residuals ``|y - y_hat|``.
        For asymmetric mode, this should contain signed residuals
        ``y - y_hat`` (positive means model under-predicted).
    predictions
        DataFrame with point forecasts in ``predicted_col``.
    coverage
        Desired coverage level (e.g. 0.9 for 90% intervals).
    residual_col
        Column name for residuals in ``cal_residuals``.
    predicted_col
        Column name for point forecasts in ``predictions``.
    id_col
        If provided, compute per-group quantiles for adaptive intervals.
    symmetric
        If ``True``, use ``|residual|`` for symmetric intervals.
        If ``False``, compute separate upper/lower bounds from signed
        residuals ``y - y_hat``.

    Returns
    -------
    pl.DataFrame
        Copy of ``predictions`` with ``y_hat_lower`` and ``y_hat_upper``
        columns appended.

    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    if residual_col not in cal_residuals.columns:
        raise ValueError(f"Column {residual_col!r} not found in cal_residuals")
    if predicted_col not in predictions.columns:
        raise ValueError(f"Column {predicted_col!r} not found in predictions")

    if symmetric:
        return _symmetric_interval(cal_residuals, predictions, coverage, residual_col, predicted_col, id_col)
    return _asymmetric_interval(cal_residuals, predictions, coverage, residual_col, predicted_col, id_col)


def _conformal_quantile(residuals: list[float], coverage: float) -> float:
    """Compute the conformal quantile with finite-sample correction.

    q_hat = ceil((n+1) * coverage) / n -th empirical quantile.
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    level = math.ceil((n + 1) * coverage) / n
    level = min(level, 1.0)
    arr = np.array(residuals, dtype=np.float64)
    return float(np.quantile(arr, level))


def _conformal_quantile_lower(residuals: list[float], alpha_half: float) -> float:
    """Compute finite-sample-corrected lower conformal quantile."""
    n = len(residuals)
    if n == 0:
        return 0.0
    level = math.floor((n + 1) * alpha_half) / n
    level = max(level, 0.0)
    arr = np.array(residuals, dtype=np.float64)
    return float(np.quantile(arr, level))


def _conformal_quantile_upper(residuals: list[float], alpha_half: float) -> float:
    """Compute finite-sample-corrected upper conformal quantile."""
    n = len(residuals)
    if n == 0:
        return 0.0
    level = math.ceil((n + 1) * (1 - alpha_half)) / n
    level = min(level, 1.0)
    arr = np.array(residuals, dtype=np.float64)
    return float(np.quantile(arr, level))


def _symmetric_interval(
    cal_residuals: pl.DataFrame,
    predictions: pl.DataFrame,
    coverage: float,
    residual_col: str,
    predicted_col: str,
    id_col: str | None,
) -> pl.DataFrame:
    """Compute symmetric conformal intervals using |residual| quantiles."""
    if id_col is not None and id_col in cal_residuals.columns:
        # Per-group conformal quantiles → join for vectorized interval computation
        group_rows: list[dict[str, Any]] = []
        for group_id, group_df in cal_residuals.group_by(id_col, maintain_order=True):
            resids = group_df[residual_col].drop_nulls().to_list()
            group_rows.append({id_col: group_id[0], "__q_hat": _conformal_quantile(resids, coverage)})

        q_df = pl.DataFrame(group_rows, schema={id_col: predictions.schema[id_col], "__q_hat": pl.Float64()})
        result = (
            predictions.join(q_df, on=id_col, how="left")
            .with_columns(
                (pl.col(predicted_col) - pl.col("__q_hat").fill_null(0.0)).alias("y_hat_lower"),
                (pl.col(predicted_col) + pl.col("__q_hat").fill_null(0.0)).alias("y_hat_upper"),
            )
            .drop("__q_hat")
        )
        return result

    # Global conformal quantile
    resids = cal_residuals[residual_col].drop_nulls().to_list()
    q_hat = _conformal_quantile(resids, coverage)
    return predictions.with_columns(
        (pl.col(predicted_col) - q_hat).alias("y_hat_lower"),
        (pl.col(predicted_col) + q_hat).alias("y_hat_upper"),
    )


def _asymmetric_interval(
    cal_residuals: pl.DataFrame,
    predictions: pl.DataFrame,
    coverage: float,
    residual_col: str,
    predicted_col: str,
    id_col: str | None,
) -> pl.DataFrame:
    """Compute asymmetric conformal intervals using signed residual quantiles."""
    alpha = 1 - coverage

    if id_col is not None and id_col in cal_residuals.columns:
        group_rows: list[dict[str, Any]] = []
        for group_id, group_df in cal_residuals.group_by(id_col, maintain_order=True):
            signed = group_df[residual_col].drop_nulls().to_list()
            lower_q = _conformal_quantile_lower(signed, alpha / 2)
            upper_q = _conformal_quantile_upper(signed, alpha / 2)
            group_rows.append({id_col: group_id[0], "__lower_q": lower_q, "__upper_q": upper_q})

        q_df = pl.DataFrame(
            group_rows,
            schema={id_col: predictions.schema[id_col], "__lower_q": pl.Float64(), "__upper_q": pl.Float64()},
        )
        result = (
            predictions.join(q_df, on=id_col, how="left")
            .with_columns(
                (pl.col(predicted_col) + pl.col("__lower_q").fill_null(0.0)).alias("y_hat_lower"),
                (pl.col(predicted_col) + pl.col("__upper_q").fill_null(0.0)).alias("y_hat_upper"),
            )
            .drop("__lower_q", "__upper_q")
        )
        return result

    signed = cal_residuals[residual_col].drop_nulls().to_list()
    lower_q = _conformal_quantile_lower(signed, alpha / 2)
    upper_q = _conformal_quantile_upper(signed, alpha / 2)
    return predictions.with_columns(
        (pl.col(predicted_col) + lower_q).alias("y_hat_lower"),
        (pl.col(predicted_col) + upper_q).alias("y_hat_upper"),
    )


class EnbPI:
    """Ensemble Batch Prediction Intervals for time series.

    A conformal method that produces adaptive prediction intervals
    using bootstrap aggregation of out-of-bag residuals. Intervals
    adapt over time via the ``update`` method.

    .. note::

       Interval width is currently constant across all forecast steps.
       In practice, prediction uncertainty grows with the horizon due to
       error accumulation in recursive prediction.

    Parameters
    ----------
    estimator_factory
        Callable returning a fresh scikit-learn-compatible estimator.
    n_bootstraps
        Number of bootstrap models to train.
    lags
        Lag offsets used as features.
    coverage
        Desired coverage level (e.g. 0.9 for 90%).
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
        n_bootstraps: int = 25,
        lags: list[int] | None = None,
        coverage: float = 0.9,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1")
        lags = lags or [1, 2, 3]
        if any(k <= 0 for k in lags):
            raise ValueError("lags must be positive integers")
        self.estimator_factory = estimator_factory
        self.n_bootstraps = n_bootstraps
        self.lags = sorted(lags)
        self.coverage = coverage
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.estimators_: list[Estimator] = []
        self.residuals_: dict[Any, list[float]] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> EnbPI:
        """Fit bootstrap ensemble and compute out-of-bag residuals.

        Parameters
        ----------
        df
            Training DataFrame.

        Returns
        -------
        EnbPI
            Fitted instance (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        rng = np.random.default_rng(42)

        # Build lag matrices per group, then pool
        group_data: list[tuple[Any, np.ndarray, np.ndarray]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            values = group_df[self.target_col].to_list()
            X, y = _build_lag_matrix(values, self.lags)
            if len(X) > 0:
                group_data.append((group_id[0], X, y))

        if not group_data:
            raise ValueError("No training samples — series are shorter than max(lags)")

        # Pool all data
        X_all = np.vstack([gd[1] for gd in group_data])
        y_all = np.concatenate([gd[2] for gd in group_data])
        n_total = len(y_all)

        # Track OOB predictions per sample
        oob_preds: list[list[float]] = [[] for _ in range(n_total)]

        self.estimators_ = []
        for _ in range(self.n_bootstraps):
            # Bootstrap sample indices
            boot_idx = rng.integers(0, n_total, size=n_total)
            oob_mask = np.ones(n_total, dtype=bool)
            oob_mask[boot_idx] = False

            est = self.estimator_factory()
            est.fit(X_all[boot_idx], y_all[boot_idx])
            self.estimators_.append(est)

            # Predict on OOB samples
            oob_indices = np.where(oob_mask)[0]
            if len(oob_indices) > 0:
                preds = est.predict(X_all[oob_indices])
                for idx, pred in zip(oob_indices, preds, strict=False):
                    oob_preds[idx].append(float(pred))

        # Compute OOB residuals per group
        self.residuals_ = {}
        offset = 0
        for gid, _X_g, y_g in group_data:
            resids: list[float] = []
            for i in range(len(y_g)):
                oob_for_sample = oob_preds[offset + i]
                if oob_for_sample:
                    agg_pred = np.mean(oob_for_sample)
                    resids.append(abs(float(y_g[i]) - agg_pred))
            offset += len(y_g)
            self.residuals_[gid] = resids

        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate *h*-step-ahead forecasts with prediction intervals.

        Parameters
        ----------
        df
            DataFrame containing the history to predict from.
        h
            Forecast horizon.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat",
            "y_hat_lower", "y_hat_upper"]``.

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
            gid = group_id[0]
            values = group_df[self.target_col].to_list()
            if len(values) < max_lag:
                raise ValueError(
                    f"Series {gid!r} has {len(values)} observations but max(lags)={max_lag} — too short to predict"
                )
            last_time = group_df[self.time_col][-1]
            buffer = list(values[-max_lag:])
            future_times = _make_future_dates(last_time, freq, h)

            # Conformal quantile from this group's OOB residuals
            resids = self.residuals_.get(gid, [])
            q_hat = _conformal_quantile(resids, self.coverage) if resids else 0.0

            for step in range(h):
                x_row = np.array([[buffer[-lag] for lag in self.lags]], dtype=np.float64)
                # Aggregate predictions from all bootstrap models
                preds = [float(est.predict(x_row)[0]) for est in self.estimators_]
                y_hat = float(np.mean(preds))
                buffer.append(y_hat)

                rows.append(
                    {
                        self.id_col: gid,
                        self.time_col: future_times[step],
                        "y_hat": y_hat,
                        "y_hat_lower": y_hat - q_hat,
                        "y_hat_upper": y_hat + q_hat,
                    }
                )

        schema: dict[str, Any] = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
            "y_hat": pl.Float64(),
            "y_hat_lower": pl.Float64(),
            "y_hat_upper": pl.Float64(),
        }
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)

    def update(self, new_obs: pl.DataFrame) -> EnbPI:
        """Update residuals with newly observed data for adaptive intervals.

        Parameters
        ----------
        new_obs
            DataFrame with actual observations to incorporate. Must
            contain ``id_col``, ``time_col``, and ``target_col``, plus
            a ``y_hat`` column with the corresponding predictions.

        Returns
        -------
        EnbPI
            Updated instance (``self``).

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before update()")

        for group_id, group_df in new_obs.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            actuals = group_df[self.target_col].to_list()
            preds = group_df["y_hat"].to_list()
            new_resids = [abs(a - p) for a, p in zip(actuals, preds, strict=False)]
            if gid not in self.residuals_:
                self.residuals_[gid] = []
            self.residuals_[gid].extend(new_resids)

        return self
