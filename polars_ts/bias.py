"""Forecast bias detection and correction. Closes #56."""

from __future__ import annotations

import numpy as np
import polars as pl


def bias_detect(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
) -> pl.DataFrame:
    """Detect systematic forecast bias.

    Computes mean error, sign test ratio, and bias ratio per group.

    Parameters
    ----------
    df
        DataFrame with actual and predicted values.
    actual_col
        Column with actual values.
    predicted_col
        Column with predicted values.
    id_col
        If provided, compute bias per group.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["mean_error", "sign_ratio", "bias_ratio"]``.
        ``sign_ratio`` is the fraction of positive errors (> 0.5 = over-predicting).
        ``bias_ratio`` is ``mean_error / MAE``.

    """
    error = pl.col(predicted_col) - pl.col(actual_col)
    abs_error = error.abs()
    positive = error.gt(0).cast(pl.Float64)

    agg_exprs = [
        error.mean().alias("mean_error"),
        positive.mean().alias("sign_ratio"),
        (error.mean() / abs_error.mean()).alias("bias_ratio"),
    ]

    if id_col is not None:
        return df.group_by(id_col).agg(agg_exprs).sort(id_col)

    return df.select(agg_exprs)


def bias_correct(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    method: str = "mean",
    id_col: str | None = None,
) -> pl.DataFrame:
    """Correct systematic forecast bias.

    Parameters
    ----------
    df
        DataFrame with actual and predicted values.
    actual_col
        Column with actual values.
    predicted_col
        Column with predicted values.
    method
        Correction method:
        ``"mean"`` — subtract mean error,
        ``"regression"`` — linear recalibration (slope + intercept),
        ``"quantile"`` — quantile mapping correction.
    id_col
        If provided, correct per group.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``predicted_col`` adjusted and
        ``{predicted_col}_original`` preserving the uncorrected values.

    """
    valid = {"mean", "regression", "quantile"}
    if method not in valid:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid)}")

    result = df.with_columns(pl.col(predicted_col).alias(f"{predicted_col}_original"))

    if method == "mean":
        if id_col is not None:
            me = pl.col(predicted_col).mean() - pl.col(actual_col).mean()
            return result.with_columns((pl.col(predicted_col) - me.over(id_col)).alias(predicted_col))
        me_val = float((df[predicted_col] - df[actual_col]).mean())  # type: ignore[arg-type]
        return result.with_columns((pl.col(predicted_col) - me_val).alias(predicted_col))

    if method == "regression":
        if id_col is not None:
            frames: list[pl.DataFrame] = []
            for _gid, group_df in result.group_by(id_col, maintain_order=True):
                corrected = _regression_correct(group_df, actual_col, predicted_col)
                frames.append(corrected)
            return pl.concat(frames)
        return _regression_correct(result, actual_col, predicted_col)

    if method == "quantile":
        if id_col is not None:
            frames = []
            for _gid, group_df in result.group_by(id_col, maintain_order=True):
                corrected = _quantile_correct(group_df, actual_col, predicted_col)
                frames.append(corrected)
            return pl.concat(frames)
        return _quantile_correct(result, actual_col, predicted_col)

    return result  # pragma: no cover


def _regression_correct(df: pl.DataFrame, actual_col: str, predicted_col: str) -> pl.DataFrame:
    """Apply linear recalibration: actual ≈ slope * predicted + intercept."""
    y = df[actual_col].to_numpy().astype(np.float64)
    x = df[predicted_col].to_numpy().astype(np.float64)
    n = len(y)
    if n < 2:
        return df

    x_mean, y_mean = x.mean(), y.mean()
    slope = float(np.dot(x - x_mean, y - y_mean) / (np.dot(x - x_mean, x - x_mean) + 1e-10))
    intercept = y_mean - slope * x_mean

    corrected = slope * x + intercept
    return df.with_columns(pl.Series(predicted_col, corrected.tolist()))


def _quantile_correct(df: pl.DataFrame, actual_col: str, predicted_col: str) -> pl.DataFrame:
    """Quantile mapping: map predicted quantiles to actual quantiles."""
    pred = np.sort(df[predicted_col].to_numpy().astype(np.float64))
    actual = np.sort(df[actual_col].to_numpy().astype(np.float64))

    raw = df[predicted_col].to_numpy().astype(np.float64)
    corrected = np.interp(raw, pred, actual)
    return df.with_columns(pl.Series(predicted_col, corrected.tolist()))
