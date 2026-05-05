"""Shared training utilities for DL forecasters."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates


def build_windows(
    arrays: list[np.ndarray],
    input_size: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window training pairs from multiple series.

    Returns
    -------
    X : np.ndarray
        Input windows of shape ``(n_windows, input_size)``.
    Y : np.ndarray
        Target windows of shape ``(n_windows, h)``.

    """
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for arr in arrays:
        arr64 = arr.astype(np.float64)
        n = len(arr64)
        for t in range(input_size, n - h + 1):
            all_x.append(arr64[t - input_size : t])
            all_y.append(arr64[t : t + h])
    if not all_x:
        return np.empty((0, input_size)), np.empty((0, h))
    return np.array(all_x), np.array(all_y)


def build_forecast_df(
    ids: list[str],
    forecasts: np.ndarray,
    df: pl.DataFrame,
    h: int,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Build output DataFrame with future dates.

    Parameters
    ----------
    ids
        Series identifiers.
    forecasts
        Array of shape ``(n_series, h)``.
    df
        Original input DataFrame.
    h
        Forecast horizon.
    id_col, time_col
        Column names.

    """
    sorted_df = df.sort(id_col, time_col)
    rows: list[dict[str, Any]] = []

    for i, sid in enumerate(ids):
        series_df = sorted_df.filter(pl.col(id_col) == sid)
        times = series_df[time_col]
        freq = _infer_freq(times)
        last_time = times[-1]
        future_dates = _make_future_dates(last_time, freq, h)

        for t in range(h):
            rows.append(
                {
                    id_col: sid,
                    time_col: future_dates[t],
                    "y_hat": float(forecasts[i, t]),
                }
            )

    return pl.DataFrame(rows)
