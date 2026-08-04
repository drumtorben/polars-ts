"""Streaming global forecaster with incremental model updates."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.streaming.window import SlidingWindowManager


class StreamingGlobalForecaster:
    """Global forecaster supporting incremental updates via partial_fit.

    Uses a scikit-learn estimator that supports ``partial_fit()``
    (e.g., SGDRegressor) for online model updates as new data arrives.

    Parameters
    ----------
    estimator
        Scikit-learn-compatible estimator with ``fit`` and ``partial_fit``.
    lags
        Lag offsets for feature engineering.
    window_size
        Maximum observations to retain per series for feature context.
    id_col, time_col, target_col
        Column name overrides.

    """

    def __init__(
        self,
        estimator: Any,
        lags: list[int],
        window_size: int = 100,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.estimator = estimator
        if any(lag <= 0 for lag in lags):
            raise ValueError("All lags must be positive integers")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.lags = sorted(lags)
        self.window_size = window_size
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self.is_fitted_ = False
        self.window_manager_: SlidingWindowManager = SlidingWindowManager(window_size)
        self.estimator_: Any = None
        self._freq: timedelta | int | float | None = None
        self._schema: dict[str, Any] = {}
        self._last_times: dict[str, Any] = {}

    def fit(self, df: pl.DataFrame) -> StreamingGlobalForecaster:
        """Batch fit: initialize window buffers and train model."""
        sorted_df = df.sort(self.id_col, self.time_col)
        self._freq = _infer_freq(sorted_df[self.time_col])
        self._schema = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
        }

        # Populate window buffers
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            sid = str(group_id[0])
            values = group_df[self.target_col].to_numpy().astype(np.float64)
            self.window_manager_.append(sid, values)
            self._last_times[sid] = group_df[self.time_col][-1]

        # Build features and fit
        X, y = self._build_features_from_windows()
        try:
            import sklearn.base
        except ImportError as e:
            raise ImportError(
                "StreamingGlobalForecaster requires scikit-learn. Install via `pip install scikit-learn`."
            ) from e

        self.estimator_ = sklearn.base.clone(self.estimator)
        self.estimator_.fit(X, y)
        self.is_fitted_ = True
        return self

    def partial_fit(self, df: pl.DataFrame) -> StreamingGlobalForecaster:
        """Incrementally update model with new observations."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before partial_fit()")

        if not hasattr(self.estimator_, "partial_fit"):
            raise TypeError(
                "Estimator does not support partial_fit(). " "Use an online-capable estimator like SGDRegressor."
            )

        sorted_df = df.sort(self.id_col, self.time_col)

        # Update window buffers with new data
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            sid = str(group_id[0])
            values = group_df[self.target_col].to_numpy().astype(np.float64)
            self.window_manager_.append(sid, values)
            self._last_times[sid] = group_df[self.time_col][-1]

        # Build features from updated windows and do incremental update
        X, y = self._build_features_from_windows()
        self.estimator_.partial_fit(X, y)
        return self

    def predict(self, h: int) -> pl.DataFrame:
        """Generate h-step forecasts using recursive prediction."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")

        assert self._freq is not None
        rows: list[dict[str, Any]] = []
        for sid in self.window_manager_.series_ids:
            values = list(self.window_manager_.get(sid))
            last_time = self._last_times[sid]
            future_times = _make_future_dates(last_time, self._freq, h)

            for step in range(h):
                features = self._make_lag_features(values)
                if features is None:
                    # Not enough history for lag features
                    pred = values[-1] if values else 0.0
                else:
                    pred = float(self.estimator_.predict(features.reshape(1, -1))[0])
                values.append(pred)
                rows.append({self.id_col: sid, self.time_col: future_times[step], "y_hat": pred})

        schema = {**self._schema, "y_hat": pl.Float64()}
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_features_from_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Build lag features from all series windows."""
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        max_lag = max(self.lags)

        for sid in self.window_manager_.series_ids:
            data = self.window_manager_.get(sid)
            if len(data) <= max_lag:
                continue

            for i in range(max_lag, len(data)):
                features = np.array([data[i - lag] for lag in self.lags])
                X_parts.append(features)
                y_parts.append(data[i])

        if not X_parts:
            return np.empty((0, len(self.lags))), np.empty(0)

        return np.vstack(X_parts), np.array(y_parts)

    def _make_lag_features(self, values: list[float]) -> np.ndarray | None:
        """Create lag feature vector from the tail of a values list."""
        if len(values) < max(self.lags):
            return None
        return np.array([values[-lag] for lag in self.lags])
