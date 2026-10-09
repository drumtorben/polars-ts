"""Intermittent-demand models for sparse perishables SKUs.

Slow movers sell on a minority of days; standard smoothers collapse to
near-zero or chase noise. Implements the classical sparse-demand family:

- **Croston** — separate exponential smoothing of nonzero demand sizes
  and inter-demand intervals.
- **SBA** (Syntetos-Boylan Approximation) — Croston with the
  ``1 - alpha/2`` bias correction.
- **TSB** (Teunter-Syntetos-Babai) — smooths demand probability every
  period, so obsolete SKUs decay toward zero.

Plus the standard ADI / CV² demand classifier (Syntetos et al.) used by
:func:`intermittent_forecast` to dispatch per SKU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.models._time_utils import _infer_freq, _make_future_dates

# Syntetos-Boylan cut-offs separating smooth/erratic from intermittent/lumpy
_ADI_CUTOFF = 1.32
_CV2_CUTOFF = 0.49


def classify_demand(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Classify each SKU's demand pattern via ADI and CV².

    ADI (average inter-demand interval, in periods) and CV² (squared
    coefficient of variation of nonzero demand sizes) split SKUs into
    the four Syntetos-Boylan quadrants at ADI 1.32 and CV² 0.49:
    ``smooth``, ``erratic``, ``intermittent``, and ``lumpy``.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "adi", "cv2", "demand_class"]``. SKUs with no
        nonzero demand get infinite ADI and class ``"intermittent"``.

    """
    nonzero = pl.col(target_col).gt(0)
    stats = (
        df.sort(id_col, time_col)
        .group_by(id_col)
        .agg(
            (pl.len() / nonzero.sum()).alias("adi"),
            ((pl.col(target_col).filter(nonzero).std(ddof=0) / pl.col(target_col).filter(nonzero).mean()) ** 2).alias(
                "cv2"
            ),
        )
        .with_columns(pl.col("cv2").fill_null(0.0))
    )
    return stats.with_columns(
        pl.when(pl.col("adi") <= _ADI_CUTOFF)
        .then(pl.when(pl.col("cv2") <= _CV2_CUTOFF).then(pl.lit("smooth")).otherwise(pl.lit("erratic")))
        .otherwise(pl.when(pl.col("cv2") <= _CV2_CUTOFF).then(pl.lit("intermittent")).otherwise(pl.lit("lumpy")))
        .alias("demand_class")
    ).sort(id_col)


def _croston_rate(values: np.ndarray, alpha: float, bias_correction: float = 1.0) -> float:
    """Return the smoothed per-period demand rate via Croston's method."""
    nonzero_idx = np.flatnonzero(values > 0)
    if nonzero_idx.size == 0:
        return 0.0
    size = float(values[nonzero_idx[0]])
    interval = float(nonzero_idx[0] + 1)
    prev = nonzero_idx[0]
    for idx in nonzero_idx[1:]:
        size += alpha * (float(values[idx]) - size)
        interval += alpha * (float(idx - prev) - interval)
        prev = idx
    return bias_correction * size / max(interval, 1.0)


def _tsb_rate(values: np.ndarray, alpha: float, beta: float) -> float:
    """Return the smoothed per-period demand rate via TSB."""
    nonzero = values > 0
    if not nonzero.any():
        return 0.0
    prob = float(nonzero.mean())
    size = float(values[nonzero][0])
    for v in values:
        if v > 0:
            prob += beta * (1.0 - prob)
            size += alpha * (float(v) - size)
        else:
            prob += beta * (0.0 - prob)
    return prob * size


def _flat_forecast(
    df: pl.DataFrame,
    h: int,
    rate_fn: Any,
    target_col: str,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Emit h rows per SKU with a constant per-period rate from rate_fn."""
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")
    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])
    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        rate = rate_fn(group_df[target_col].to_numpy())
        for t in _make_future_dates(group_df[time_col][-1], freq, h):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": rate})
    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def croston_forecast(
    df: pl.DataFrame,
    h: int,
    alpha: float = 0.1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Croston's method: constant demand-rate forecast for sparse SKUs.

    Parameters
    ----------
    df
        Gap-free canonical sales frame (zeros must be explicit rows).
    h
        Forecast horizon (number of periods).
    alpha
        Smoothing factor for both demand sizes and intervals.
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]`` with *h* rows per SKU;
        ``y_hat`` is the expected demand per period.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return _flat_forecast(df, h, lambda v: _croston_rate(v, alpha), target_col, id_col, time_col)


def sba_forecast(
    df: pl.DataFrame,
    h: int,
    alpha: float = 0.1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Syntetos-Boylan Approximation: bias-corrected Croston.

    Applies the ``1 - alpha/2`` correction that removes Croston's
    positive bias; the recommended default for intermittent and lumpy
    SKUs.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    h
        Forecast horizon.
    alpha
        Smoothing factor.
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]``.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    correction = 1.0 - alpha / 2.0
    return _flat_forecast(df, h, lambda v: _croston_rate(v, alpha, correction), target_col, id_col, time_col)


def tsb_forecast(
    df: pl.DataFrame,
    h: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Teunter-Syntetos-Babai forecast for sparse SKUs with obsolescence.

    Unlike Croston, the demand probability updates every period, so a
    SKU that stops selling sees its forecast decay toward zero — the
    right behaviour for delisted or seasonal perishables.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    h
        Forecast horizon.
    alpha
        Smoothing factor for demand sizes.
    beta
        Smoothing factor for demand probability.
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]``.

    """
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must be in (0, 1)")
    return _flat_forecast(df, h, lambda v: _tsb_rate(v, alpha, beta), target_col, id_col, time_col)


def intermittent_forecast(
    df: pl.DataFrame,
    h: int,
    method: str = "auto",
    alpha: float = 0.1,
    beta: float = 0.1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Forecast sparse SKUs, optionally auto-dispatching per demand class.

    With ``method="auto"``, each SKU is classified via
    :func:`classify_demand`: intermittent and lumpy SKUs use SBA,
    smooth and erratic SKUs use Croston.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    h
        Forecast horizon.
    method
        ``"auto"``, ``"croston"``, ``"sba"``, or ``"tsb"``.
    alpha, beta
        Smoothing factors (``beta`` only used by TSB).
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]``.

    """
    dispatch = {"croston": croston_forecast, "sba": sba_forecast}
    if method in dispatch:
        return dispatch[method](df, h, alpha, target_col, id_col, time_col)
    if method == "tsb":
        return tsb_forecast(df, h, alpha, beta, target_col, id_col, time_col)
    if method != "auto":
        raise ValueError(f"Unknown method {method!r}; expected 'auto', 'croston', 'sba', or 'tsb'")

    classes = classify_demand(df, target_col, id_col, time_col)
    sparse_ids = classes.filter(pl.col("demand_class").is_in(["intermittent", "lumpy"]))[id_col].to_list()
    parts = []
    sparse = df.filter(pl.col(id_col).is_in(sparse_ids))
    dense = df.filter(~pl.col(id_col).is_in(sparse_ids))
    if sparse.height:
        parts.append(sba_forecast(sparse, h, alpha, target_col, id_col, time_col))
    if dense.height:
        parts.append(croston_forecast(dense, h, alpha, target_col, id_col, time_col))
    return pl.concat(parts).sort(id_col, time_col)
