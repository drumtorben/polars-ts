"""End-to-end ML forecasting pipeline.

Ties feature engineering and target transforms into a single
fit/predict interface.  Implements Ch 8 of "Modern Time Series
Forecasting with Python" (2nd Ed.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.models.multistep import Estimator

# ---------------------------------------------------------------------------
# Shared helpers (also imported by global_model.py)
# ---------------------------------------------------------------------------


def _build_feature_df(
    df: pl.DataFrame,
    lags: list[int] | None,
    rolling_windows: list[int] | None,
    rolling_aggs: list[str] | None,
    calendar: list[str] | None,
    fourier: list[tuple[float, int]] | None,
    target_col: str,
    id_col: str,
    time_col: str,
    past_covariates: list[str] | None = None,
    past_covariate_lags: list[int] | None = None,
    future_covariates: list[str] | None = None,
) -> pl.DataFrame:
    """Apply all configured feature engineering steps to *df*."""
    result = df
    if lags:
        from polars_ts.features.lags import lag_features

        result = lag_features(result, lags, target_col, id_col, time_col)
    if past_covariates:
        from polars_ts.features.lags import covariate_lag_features

        cov_lags = past_covariate_lags or lags or [1]
        result = covariate_lag_features(result, past_covariates, cov_lags, id_col, time_col)
    if rolling_windows:
        from polars_ts.features.rolling import rolling_features

        result = rolling_features(result, rolling_windows, rolling_aggs, target_col, id_col, time_col)
    if calendar:
        from polars_ts.features.calendar import calendar_features

        result = calendar_features(result, calendar, time_col)
    if fourier:
        from polars_ts.features.fourier import fourier_features

        for period, n_harmonics in fourier:
            result = fourier_features(result, period, n_harmonics, time_col, id_col)
    # Future covariates are already columns in df — move them to the end
    # so feature column order matches _build_step_features output order.
    if future_covariates:
        other_cols = [c for c in result.columns if c not in future_covariates]
        result = result.select(other_cols + future_covariates)
    return result


def _apply_transform(
    df: pl.DataFrame,
    transform: str | None,
    kwargs: dict[str, Any] | None,
    target_col: str,
    id_col: str,
    time_col: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply a target transform and return (transformed_df, state)."""
    if transform is None:
        return df, {}

    kw = kwargs or {}

    if transform == "log":
        from polars_ts.transforms.log import log_transform

        return log_transform(df, target_col), {"type": "log"}

    if transform == "boxcox":
        from polars_ts.transforms.boxcox import boxcox_transform

        lam = kw.get("lam", 1.0)
        return boxcox_transform(df, lam, target_col), {"type": "boxcox", "lam": lam}

    if transform == "difference":
        from polars_ts.transforms.differencing import difference

        order = kw.get("order", 1)
        period = kw.get("period", 1)
        result = difference(df, order, period, target_col, id_col, time_col)
        # Store last values per group for undifferencing during predict
        initial_col = f"{target_col}_diff_initial"
        initials: dict[Any, list[float]] = {}
        if initial_col in result.columns:
            for group_id, group_df in result.group_by(id_col, maintain_order=True):
                vals = group_df[initial_col][0]
                if isinstance(vals, list):
                    initials[group_id[0]] = vals
                else:
                    initials[group_id[0]] = [vals]
        return result, {"type": "difference", "order": order, "period": period, "initials": initials}

    raise ValueError(f"Unknown target_transform: {transform!r}")


def _build_step_features(
    buffer: list[float],
    timestamp: Any,
    lags: list[int] | None,
    rolling_windows: list[int] | None,
    rolling_aggs: list[str] | None,
    calendar: list[str] | None,
    fourier_specs: list[tuple[float, int]] | None,
    step_index: int,
    _time_col: str,
    past_covariate_buffers: dict[str, list[float]] | None = None,
    past_covariate_lags: list[int] | None = None,
    future_covariate_values: dict[str, float] | None = None,
) -> list[float]:
    """Build a single feature vector for one recursive prediction step."""
    features: list[float] = []

    # Lag features
    if lags:
        for lag in sorted(lags):
            features.append(buffer[-lag])

    # Past covariate lag features
    if past_covariate_buffers and past_covariate_lags:
        for col in sorted(past_covariate_buffers.keys()):
            buf = past_covariate_buffers[col]
            for lag in sorted(past_covariate_lags):
                if lag <= len(buf):
                    features.append(buf[-lag])
                else:
                    features.append(buf[0] if buf else 0.0)

    # Rolling features (must match polars_ts.features.rolling._DEFAULT_AGGS)
    if rolling_windows:
        aggs = rolling_aggs or ["mean", "std", "min", "max"]
        for w in rolling_windows:
            window = buffer[-w:] if len(buffer) >= w else buffer[:]
            arr = np.array(window, dtype=np.float64)
            for agg in aggs:
                if agg == "mean":
                    features.append(float(np.mean(arr)))
                elif agg == "std":
                    features.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
                elif agg == "min":
                    features.append(float(np.min(arr)))
                elif agg == "max":
                    features.append(float(np.max(arr)))
                elif agg == "sum":
                    features.append(float(np.sum(arr)))
                elif agg == "median":
                    features.append(float(np.median(arr)))
                elif agg == "var":
                    features.append(float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0)

    # Calendar features
    if calendar and hasattr(timestamp, "weekday"):
        _cal_map = {
            "day_of_week": lambda t: float(t.isoweekday() % 7 + 1) if hasattr(t, "isoweekday") else 0.0,
            "day_of_month": lambda t: float(t.day),
            "day_of_year": lambda t: float(t.timetuple().tm_yday) if hasattr(t, "timetuple") else 0.0,
            "week": lambda t: float(t.isocalendar()[1]) if hasattr(t, "isocalendar") else 0.0,
            "month": lambda t: float(t.month),
            "quarter": lambda t: float((t.month - 1) // 3 + 1),
            "year": lambda t: float(t.year),
            "hour": lambda t: float(t.hour) if hasattr(t, "hour") else 0.0,
            "minute": lambda t: float(t.minute) if hasattr(t, "minute") else 0.0,
            "is_weekend": lambda t: float(t.isoweekday() >= 6) if hasattr(t, "isoweekday") else 0.0,
        }
        for feat_name in sorted(calendar):
            if feat_name in _cal_map:
                features.append(_cal_map[feat_name](timestamp))

    # Fourier features
    if fourier_specs:
        for period, n_harmonics in fourier_specs:
            for k in range(1, n_harmonics + 1):
                angle = 2 * np.pi * k * step_index / period
                features.append(float(np.sin(angle)))
                features.append(float(np.cos(angle)))

    # Future covariate values (appended last to match column order)
    if future_covariate_values:
        for col in sorted(future_covariate_values.keys()):
            features.append(future_covariate_values[col])

    return features


# ---------------------------------------------------------------------------
# ForecastPipeline
# ---------------------------------------------------------------------------


class ForecastPipeline:
    """End-to-end ML forecasting pipeline with feature engineering and transforms.

    Combines lag features, rolling aggregations, calendar features,
    Fourier terms, and optional target transforms into a single
    fit/predict workflow.

    Parameters
    ----------
    estimator
        A scikit-learn-compatible estimator with ``fit`` and ``predict``.
    lags
        Lag offsets for lag features (e.g. ``[1, 2, 7]``).
    rolling_windows
        Window sizes for rolling aggregations (e.g. ``[7, 14]``).
    rolling_aggs
        Aggregation functions for rolling features (default ``["mean"]``).
    calendar
        Calendar features to extract (e.g. ``["day_of_week", "month"]``).
    fourier
        Fourier term specs as ``[(period, n_harmonics), ...]``.
    target_transform
        Optional transform: ``"log"``, ``"boxcox"``, or ``"difference"``.
    transform_kwargs
        Arguments passed to the transform function (e.g. ``{"lam": 0.5}``).
    past_covariates
        Column names of time-varying covariates known only up to the
        present.  Lagged features are created automatically.
    future_covariates
        Column names of time-varying covariates known into the future
        (e.g. holidays, promotions).  Values for the forecast horizon
        must be supplied in *future_df* at predict time.
    past_covariate_lags
        Lag offsets for past covariate features.  Defaults to *lags*.
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        estimator: Estimator,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        rolling_aggs: list[str] | None = None,
        calendar: list[str] | None = None,
        fourier: list[tuple[float, int]] | None = None,
        target_transform: str | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        past_covariates: list[str] | None = None,
        future_covariates: list[str] | None = None,
        past_covariate_lags: list[int] | None = None,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not any([lags, rolling_windows, calendar, fourier]):
            raise ValueError(
                "At least one feature source must be configured (lags, rolling_windows, calendar, fourier)"
            )
        self.estimator = estimator
        self.lags = sorted(lags) if lags else None
        self.rolling_windows = sorted(rolling_windows) if rolling_windows else None
        self.rolling_aggs = rolling_aggs
        self.calendar = sorted(calendar) if calendar else None
        self.fourier = fourier
        self.target_transform = target_transform
        self.transform_kwargs = transform_kwargs or {}
        self.past_covariates = sorted(past_covariates) if past_covariates else None
        self.future_covariates = sorted(future_covariates) if future_covariates else None
        self.past_covariate_lags = sorted(past_covariate_lags) if past_covariate_lags else None
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.is_fitted_: bool = False
        self.feature_columns_: list[str] = []
        self.transform_state_: dict[str, Any] = {}
        self.train_tail_: dict[Any, list[float]] = {}
        self.past_cov_tail_: dict[Any, dict[str, list[float]]] = {}

    def fit(self, df: pl.DataFrame) -> ForecastPipeline:
        """Fit the pipeline: transform target, build features, train model.

        Parameters
        ----------
        df
            Training DataFrame with ``id_col``, ``time_col``,
            ``target_col``, and any covariate columns.

        Returns
        -------
        ForecastPipeline
            Fitted pipeline (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)

        # Store per-series tail for recursive prediction seeding
        cov_lags = self.past_covariate_lags or self.lags or []
        buf_size = max(
            max(self.lags) if self.lags else 0,
            max(self.rolling_windows) if self.rolling_windows else 0,
            max(cov_lags) if cov_lags else 0,
        )
        buf_size = max(buf_size, 1)
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            self.train_tail_[gid] = group_df[self.target_col].tail(buf_size).to_list()
            if self.past_covariates:
                self.past_cov_tail_[gid] = {col: group_df[col].tail(buf_size).to_list() for col in self.past_covariates}

        # Apply target transform
        transformed, self.transform_state_ = _apply_transform(
            sorted_df, self.target_transform, self.transform_kwargs, self.target_col, self.id_col, self.time_col
        )

        # Track original columns so we can exclude non-feature columns later
        original_cols = set(transformed.columns)

        # Build features
        featured = _build_feature_df(
            transformed,
            self.lags,
            self.rolling_windows,
            self.rolling_aggs,
            self.calendar,
            self.fourier,
            self.target_col,
            self.id_col,
            self.time_col,
            past_covariates=self.past_covariates,
            past_covariate_lags=self.past_covariate_lags,
            future_covariates=self.future_covariates,
        )

        # Determine feature columns: generated columns + future covariates only
        generated = set(featured.columns) - original_cols
        future_cov_set = set(self.future_covariates) if self.future_covariates else set()
        self.feature_columns_ = [c for c in featured.columns if c in generated or c in future_cov_set]

        # Drop nulls from feature construction
        clean = featured.drop_nulls(subset=self.feature_columns_)

        X = clean.select(self.feature_columns_).to_numpy().astype(np.float64)
        y = clean[self.target_col].to_numpy().astype(np.float64)

        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        df: pl.DataFrame,
        h: int,
        future_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Generate *h*-step-ahead forecasts using recursive prediction.

        Parameters
        ----------
        df
            DataFrame containing history to predict from.
        h
            Forecast horizon.
        future_df
            DataFrame with future covariate values for the forecast
            horizon.  Required when *future_covariates* were specified
            at construction.  Must contain ``id_col``, ``time_col``,
            and all future covariate columns.

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

        # Build future covariate lookup: {(gid, time) -> {col: val}}
        future_lookup: dict[tuple[Any, Any], dict[str, float]] = {}
        if self.future_covariates and future_df is not None:
            for row in future_df.iter_rows(named=True):
                key = (row[self.id_col], row[self.time_col])
                future_lookup[key] = {col: float(row[col]) for col in self.future_covariates}

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = group_df[self.target_col].to_list()
            # If transform is differencing, work in the transformed space
            if self.target_transform == "difference":
                order = self.transform_state_.get("order", 1)
                period = self.transform_state_.get("period", 1)
                # Difference the values for the buffer
                diff_values = list(values)
                for _ in range(order):
                    diff_values = [
                        diff_values[i] - diff_values[i - period] if i >= period else float("nan")
                        for i in range(len(diff_values))
                    ]
                buffer = [v for v in diff_values if not (isinstance(v, float) and np.isnan(v))]
            else:
                if self.target_transform == "log":
                    buffer = [float(np.log1p(v)) for v in values]
                elif self.target_transform == "boxcox":
                    lam = self.transform_state_.get("lam", 1.0)
                    if lam == 0:
                        buffer = [float(np.log(v)) for v in values]
                    else:
                        buffer = [float((v**lam - 1) / lam) for v in values]
                else:
                    buffer = list(values)

            last_time = group_df[self.time_col][-1]
            future_times = _make_future_dates(last_time, freq, h)
            # Track original-scale values for undifferencing
            orig_values = list(values)

            # Past covariate buffers for this series
            pcov_buffers: dict[str, list[float]] | None = None
            if self.past_covariates:
                pcov_buffers = {}
                for col in self.past_covariates:
                    pcov_buffers[col] = group_df[col].to_list()

            cov_lags = self.past_covariate_lags or self.lags

            for step in range(h):
                # Future covariate values for this step
                fcov_vals: dict[str, float] | None = None
                if self.future_covariates:
                    key = (gid, future_times[step])
                    fcov_vals = future_lookup.get(key, {col: 0.0 for col in self.future_covariates})

                feat = _build_step_features(
                    buffer,
                    future_times[step],
                    self.lags,
                    self.rolling_windows,
                    self.rolling_aggs,
                    self.calendar,
                    self.fourier,
                    step,
                    self.time_col,
                    past_covariate_buffers=pcov_buffers,
                    past_covariate_lags=cov_lags,
                    future_covariate_values=fcov_vals,
                )
                x_row = np.array([feat], dtype=np.float64)
                pred_transformed = float(self.estimator.predict(x_row)[0])
                buffer.append(pred_transformed)

                # Inverse transform to original scale
                pred = self._inverse_single(pred_transformed, orig_values)
                orig_values.append(pred)

                rows.append({self.id_col: gid, self.time_col: future_times[step], "y_hat": pred})

        schema: dict[str, Any] = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
            "y_hat": pl.Float64(),
        }
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)

    def _inverse_single(self, pred: float, orig_values: list[float]) -> float:
        """Inverse-transform a single prediction to original scale."""
        if self.target_transform == "log":
            return float(np.expm1(pred))
        if self.target_transform == "boxcox":
            lam = self.transform_state_.get("lam", 1.0)
            if lam == 0:
                return float(np.exp(pred))
            return float((pred * lam + 1) ** (1.0 / lam))
        if self.target_transform == "difference":
            period = self.transform_state_.get("period", 1)
            # y_t = diff_t + y_{t-period}
            if len(orig_values) >= period:
                return pred + orig_values[-period]
            return pred
        return pred
