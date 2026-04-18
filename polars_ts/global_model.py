"""Global forecasting model — one model across all series.

Implements Ch 10 of "Modern Time Series Forecasting with Python"
(2nd Ed.).  A global model pools data from all time series and
optionally encodes series identity and static features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.models.multistep import Estimator
from polars_ts.pipeline import (
    _apply_transform,
    _build_feature_df,
    _build_step_features,
)


class GlobalForecaster:
    """Train a single ML model across all time series in a panel.

    Extends :class:`ForecastPipeline` with cross-series learning,
    optional series-identity encoding, and static exogenous features.

    Parameters
    ----------
    estimator
        A scikit-learn-compatible estimator with ``fit`` and ``predict``.
    lags
        Lag offsets for lag features.
    rolling_windows
        Window sizes for rolling aggregations.
    rolling_aggs
        Aggregation functions for rolling features.
    calendar
        Calendar features to extract.
    fourier
        Fourier term specs as ``[(period, n_harmonics), ...]``.
    target_transform
        Optional transform: ``"log"``, ``"boxcox"``, or ``"difference"``.
    transform_kwargs
        Arguments passed to the transform function.
    encode_id
        How to encode series identity:
        ``"ordinal"`` maps each ID to an integer,
        ``"onehot"`` creates binary indicator columns.
        ``None`` (default) does not encode the ID.
    static_features
        Column names of static (per-series) exogenous features to
        include in the feature matrix (e.g. ``["store_type"]``).
        String columns are ordinal-encoded automatically.
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
        encode_id: str | None = None,
        static_features: list[str] | None = None,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if not any([lags, rolling_windows, calendar, fourier]):
            raise ValueError("At least one feature source must be configured")
        if encode_id is not None and encode_id not in ("ordinal", "onehot"):
            raise ValueError(f"encode_id must be 'ordinal', 'onehot', or None, got {encode_id!r}")
        self.estimator = estimator
        self.lags = sorted(lags) if lags else None
        self.rolling_windows = sorted(rolling_windows) if rolling_windows else None
        self.rolling_aggs = rolling_aggs
        self.calendar = sorted(calendar) if calendar else None
        self.fourier = fourier
        self.target_transform = target_transform
        self.transform_kwargs = transform_kwargs or {}
        self.encode_id = encode_id
        self.static_features = static_features
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.is_fitted_: bool = False
        self.feature_columns_: list[str] = []
        self.transform_state_: dict[str, Any] = {}
        self.train_tail_: dict[Any, list[float]] = {}
        self.id_encoder_: dict[Any, int] = {}
        self.id_categories_: list[Any] = []
        self.static_encoders_: dict[str, dict[Any, int]] = {}
        self.static_values_: dict[Any, dict[str, Any]] = {}

    def fit(self, df: pl.DataFrame) -> GlobalForecaster:
        """Fit a single model on pooled data from all time series.

        Parameters
        ----------
        df
            Training DataFrame with ``id_col``, ``time_col``, ``target_col``,
            and any ``static_features`` columns.

        Returns
        -------
        GlobalForecaster
            Fitted model (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)

        # Learn ID encoding
        self.id_categories_ = sorted(sorted_df[self.id_col].unique().to_list())
        self.id_encoder_ = {cat: i for i, cat in enumerate(self.id_categories_)}

        # Store static feature values and learn encoders for string columns
        if self.static_features:
            for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
                gid = group_id[0]
                self.static_values_[gid] = {col: group_df[col][0] for col in self.static_features}
            for col in self.static_features:
                vals = sorted_df[col]
                if vals.dtype == pl.Utf8 or vals.dtype == pl.String:
                    unique_vals = sorted(vals.unique().to_list())
                    self.static_encoders_[col] = {v: i for i, v in enumerate(unique_vals)}

        # Store per-series tail
        buf_size = max(
            max(self.lags) if self.lags else 0,
            max(self.rolling_windows) if self.rolling_windows else 0,
        )
        buf_size = max(buf_size, 1)
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            self.train_tail_[group_id[0]] = group_df[self.target_col].tail(buf_size).to_list()

        # Apply target transform
        transformed, self.transform_state_ = _apply_transform(
            sorted_df,
            self.target_transform,
            self.transform_kwargs,
            self.target_col,
            self.id_col,
            self.time_col,
        )

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
        )

        # Add ID encoding
        featured = self._encode_ids(featured)

        # Add static features
        featured = self._encode_static(featured)

        # Determine feature columns
        exclude = {self.id_col, self.time_col, self.target_col}
        exclude |= {
            c
            for c in featured.columns
            if c.endswith("_original") or c.endswith("_boxcox_lambda") or c.endswith("_diff_initial")
        }
        if self.static_features:
            exclude |= set(self.static_features)
        self.feature_columns_ = [c for c in featured.columns if c not in exclude]

        # Drop nulls and fit
        clean = featured.drop_nulls(subset=[c for c in self.feature_columns_ if not c.startswith("__id_")])
        X = clean.select(self.feature_columns_).to_numpy().astype(np.float64)
        y = clean[self.target_col].to_numpy().astype(np.float64)

        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate *h*-step-ahead forecasts for all series.

        Parameters
        ----------
        df
            DataFrame containing history to predict from. Must include
            any ``static_features`` columns.
        h
            Forecast horizon.

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

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = group_df[self.target_col].to_list()

            # Build buffer in transformed space
            buffer = self._transform_buffer(values)
            last_time = group_df[self.time_col][-1]
            future_times = _make_future_dates(last_time, freq, h)
            orig_values = list(values)

            # Extra features from series identity and static features
            extra = self._build_extra_features(gid)

            for step in range(h):
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
                )
                feat.extend(extra)
                x_row = np.array([feat], dtype=np.float64)
                pred_transformed = float(self.estimator.predict(x_row)[0])
                buffer.append(pred_transformed)

                pred = self._inverse_single(pred_transformed, orig_values)
                orig_values.append(pred)

                rows.append({self.id_col: gid, self.time_col: future_times[step], "y_hat": pred})

        schema: dict[str, Any] = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
            "y_hat": pl.Float64(),
        }
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)

    # ---- private helpers ----

    def _encode_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.encode_id == "ordinal":
            mapping = self.id_encoder_
            encoded = df[self.id_col].map_elements(lambda x: mapping.get(x, -1), return_dtype=pl.Int64)
            return df.with_columns(encoded.alias("__id_encoded"))
        if self.encode_id == "onehot":
            for cat in self.id_categories_:
                col_name = f"__id_{cat}"
                df = df.with_columns((pl.col(self.id_col) == cat).cast(pl.Float64).alias(col_name))
            return df
        return df

    def _encode_static(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.static_features:
            return df
        for col in self.static_features:
            if col in self.static_encoders_:
                enc_map: dict[Any, int] = self.static_encoders_[col]
                encoded = df[col].map_elements(lambda x, m=enc_map: m.get(x, -1), return_dtype=pl.Int64)
                df = df.with_columns(encoded.alias(f"__static_{col}").cast(pl.Float64))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(f"__static_{col}"))
        return df

    def _build_extra_features(self, gid: Any) -> list[float]:
        """Build the identity + static feature vector for a series."""
        extra: list[float] = []
        if self.encode_id == "ordinal":
            extra.append(float(self.id_encoder_.get(gid, -1)))
        elif self.encode_id == "onehot":
            for cat in self.id_categories_:
                extra.append(1.0 if cat == gid else 0.0)

        if self.static_features:
            static_vals = self.static_values_.get(gid, {})
            for col in self.static_features:
                val = static_vals.get(col)
                if col in self.static_encoders_:
                    extra.append(float(self.static_encoders_[col].get(val, -1)))
                else:
                    extra.append(float(val) if val is not None else 0.0)

        return extra

    def _transform_buffer(self, values: list[float]) -> list[float]:
        """Transform raw values into the model's working space."""
        if self.target_transform == "log":
            return [float(np.log1p(v)) for v in values]
        if self.target_transform == "boxcox":
            lam = self.transform_state_.get("lam", 1.0)
            if lam == 0:
                return [float(np.log(v)) for v in values]
            return [float((v**lam - 1) / lam) for v in values]
        if self.target_transform == "difference":
            period = self.transform_state_.get("period", 1)
            order = self.transform_state_.get("order", 1)
            result = list(values)
            for _ in range(order):
                result = [result[i] - result[i - period] if i >= period else float("nan") for i in range(len(result))]
            return [v for v in result if not (isinstance(v, float) and np.isnan(v))]
        return list(values)

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
            if len(orig_values) >= period:
                return pred + orig_values[-period]
            return pred
        return pred
