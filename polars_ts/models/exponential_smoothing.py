"""Pure-Polars exponential smoothing forecasters.

Implements SES, Holt's linear, and Holt-Winters methods without
external dependencies. Closes #49.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates


def ses_forecast(
    df: pl.DataFrame,
    h: int,
    alpha: float = 0.3,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Forecast with Simple Exponential Smoothing.

    Smooths with level parameter *alpha* and projects a flat forecast.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    alpha
        Smoothing parameter for level (0 < alpha < 1).

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = group_df[target_col].to_list()
        last_time = group_df[time_col][-1]

        # Initialize level with first value
        level = float(values[0])
        for v in values[1:]:
            level = alpha * float(v) + (1 - alpha) * level

        future_times = _make_future_dates(last_time, freq, h)
        for t in future_times:
            rows.append({id_col: group_id[0], time_col: t, "y_hat": level})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def holt_forecast(
    df: pl.DataFrame,
    h: int,
    alpha: float = 0.3,
    beta: float = 0.1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Holt's linear trend forecast.

    Smooths level and trend, then extrapolates linearly.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    alpha
        Smoothing parameter for level.
    beta
        Smoothing parameter for trend.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = group_df[target_col].to_list()
        last_time = group_df[time_col][-1]

        if len(values) < 2:
            raise ValueError(f"Series {group_id[0]!r} needs at least 2 observations for Holt's method")

        level = float(values[0])
        trend = float(values[1]) - float(values[0])

        for v in values[1:]:
            prev_level = level
            level = alpha * float(v) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        future_times = _make_future_dates(last_time, freq, h)
        for step, t in enumerate(future_times, start=1):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": level + step * trend})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)


def holt_winters_forecast(
    df: pl.DataFrame,
    h: int,
    season_length: int,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.1,
    seasonal: str = "additive",
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Holt-Winters seasonal forecast.

    Smooths level, trend, and seasonal components.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    season_length
        Number of observations per season.
    alpha
        Smoothing parameter for level.
    beta
        Smoothing parameter for trend.
    gamma
        Smoothing parameter for seasonality.
    seasonal
        ``"additive"`` or ``"multiplicative"``.

    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if season_length < 2:
        raise ValueError("season_length must be >= 2")
    if seasonal not in ("additive", "multiplicative"):
        raise ValueError(f"seasonal must be 'additive' or 'multiplicative', got {seasonal!r}")
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = [float(v) for v in group_df[target_col].to_list()]
        last_time = group_df[time_col][-1]
        m = season_length

        if len(values) < 2 * m:
            raise ValueError(
                f"Series {group_id[0]!r} needs at least 2*season_length={2 * m} " f"observations, got {len(values)}"
            )

        # Initialize: average of first season for level
        first_season_avg = sum(values[:m]) / m
        level = first_season_avg
        trend = (sum(values[m : 2 * m]) / m - first_season_avg) / m

        if seasonal == "additive":
            seasons = [values[i] - first_season_avg for i in range(m)]
        else:
            seasons = [values[i] / first_season_avg if first_season_avg != 0 else 1.0 for i in range(m)]

        # Smooth through all observations
        for t in range(m, len(values)):
            v = values[t]
            s_idx = t % m
            prev_level = level

            if seasonal == "additive":
                level = alpha * (v - seasons[s_idx]) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasons[s_idx] = gamma * (v - level) + (1 - gamma) * seasons[s_idx]
            else:
                level = alpha * (v / seasons[s_idx] if seasons[s_idx] != 0 else v) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasons[s_idx] = gamma * (v / level if level != 0 else 1.0) + (1 - gamma) * seasons[s_idx]

        future_times = _make_future_dates(last_time, freq, h)
        for step, ft in enumerate(future_times, start=1):
            s_idx = (len(values) - 1 + step) % m
            if seasonal == "additive":
                forecast = level + step * trend + seasons[s_idx]
            else:
                forecast = (level + step * trend) * seasons[s_idx]
            rows.append({id_col: group_id[0], time_col: ft, "y_hat": forecast})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)
