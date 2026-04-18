"""Baseline forecast models for time series benchmarking.

Implements naive, seasonal naive, moving average, and FFT-based forecasts
from Ch 4 of "Modern Time Series Forecasting with Python" (2nd Ed.).
These serve as simple benchmarks against which more complex models are compared.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def _infer_freq(times: pl.Series) -> timedelta:
    """Infer the time frequency from a sorted datetime/date series."""
    if len(times) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency")
    diffs = times.diff().drop_nulls()
    if diffs.dtype == pl.Duration:
        return diffs.median()  # type: ignore[return-value]
    # Date column → cast to duration via subtraction
    casted = times.cast(pl.Datetime("ms"))
    diffs = casted.diff().drop_nulls()
    return diffs.median()  # type: ignore[return-value]


def _make_future_dates(last_time: Any, freq: timedelta, h: int) -> list[Any]:
    """Generate h future timestamps starting from last_time + freq."""
    return [last_time + freq * (i + 1) for i in range(h)]


def naive_forecast(
    df: pl.DataFrame,
    h: int,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Naive forecast: repeat the last observed value for h steps.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    h
        Forecast horizon (number of steps ahead).
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, time_col, "y_hat"]`` containing
        *h* forecast rows per series.

    """
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    # Get last value and last timestamp per group
    last = sorted_df.group_by(id_col).agg(
        pl.col(target_col).last().alias("__last_val"),
        pl.col(time_col).last().alias("__last_time"),
    )

    rows: list[dict[str, Any]] = []
    for row in last.iter_rows(named=True):
        future_times = _make_future_dates(row["__last_time"], freq, h)
        for t in future_times:
            rows.append({id_col: row[id_col], time_col: t, "y_hat": row["__last_val"]})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def seasonal_naive_forecast(
    df: pl.DataFrame,
    h: int,
    season_length: int,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Seasonal naive forecast: repeat the last season's values cyclically.

    For each forecast step *i*, the prediction is the observed value from
    *season_length* steps before the end of the series, cycling through the
    last full season.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    h
        Forecast horizon.
    season_length
        Number of observations per season (e.g. 7 for daily data with
        weekly seasonality, 12 for monthly with yearly seasonality).
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, time_col, "y_hat"]``.

    """
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")
    if season_length <= 0:
        raise ValueError("season_length must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = group_df[target_col].to_list()
        last_time = group_df[time_col][-1]
        # Take the last season_length values (or all if fewer)
        season = values[-season_length:]
        future_times = _make_future_dates(last_time, freq, h)
        for i, t in enumerate(future_times):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": float(season[i % len(season)])})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def moving_average_forecast(
    df: pl.DataFrame,
    h: int,
    window_size: int,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Forecast the mean of the last *window_size* observed values.

    The same average is repeated for all *h* forecast steps (flat forecast).

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    h
        Forecast horizon.
    window_size
        Number of most recent observations to average.
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, time_col, "y_hat"]``.

    """
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = group_df[target_col].to_list()
        window = values[-window_size:]
        avg = sum(window) / len(window)
        last_time = group_df[time_col][-1]
        future_times = _make_future_dates(last_time, freq, h)
        for t in future_times:
            rows.append({id_col: group_id[0], time_col: t, "y_hat": avg})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def fft_forecast(
    df: pl.DataFrame,
    h: int,
    n_harmonics: int = 5,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """FFT-based forecast using dominant frequency components.

    Decomposes the series via FFT, keeps the top *n_harmonics* frequency
    components, and extrapolates them forward.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    h
        Forecast horizon.
    n_harmonics
        Number of dominant harmonics to retain. More harmonics capture
        finer detail but risk overfitting noise.
    target_col
        Column with the target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, time_col, "y_hat"]``.

    """
    import numpy as np

    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")
    if n_harmonics <= 0:
        raise ValueError("n_harmonics must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        y = np.array(group_df[target_col].to_list(), dtype=np.float64)
        n = len(y)
        last_time = group_df[time_col][-1]

        # FFT decomposition
        fft_vals = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n)

        # Keep only top n_harmonics by magnitude (plus DC component)
        magnitudes = np.abs(fft_vals)
        # DC is index 0 — always keep it; pick top harmonics from the rest
        k = min(n_harmonics, len(magnitudes) - 1)
        top_indices = np.argsort(magnitudes[1:])[-k:] + 1
        mask = np.zeros_like(fft_vals)
        mask[0] = fft_vals[0]
        mask[top_indices] = fft_vals[top_indices]

        # Reconstruct and extrapolate
        future_times = _make_future_dates(last_time, freq, h)
        for step in range(h):
            t = n + step
            val = mask[0].real / n  # DC component
            for idx in top_indices:
                val += 2 * np.abs(mask[idx]) / n * np.cos(2 * np.pi * freqs[idx] * t + np.angle(mask[idx]))
            rows.append({id_col: group_id[0], time_col: future_times[step], "y_hat": float(val)})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)
