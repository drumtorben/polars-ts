"""Pure-Polars exponential smoothing forecasters.

Implements SES, Holt's linear, and Holt-Winters methods.
Delegates to Rust when available (2-5x faster per group),
falling back to pure Python otherwise. Closes #49.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates

# ---------------------------------------------------------------------------
# Python fallbacks
# ---------------------------------------------------------------------------


def _ses_python(values: list[float], alpha: float, h: int) -> list[float]:
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return [level] * h


def _holt_python(values: list[float], alpha: float, beta: float, h: int) -> list[float]:
    level = values[0]
    trend = values[1] - values[0]
    for v in values[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return [level + step * trend for step in range(1, h + 1)]


def _hw_python(
    values: list[float], alpha: float, beta: float, gamma: float, m: int, additive: bool, h: int
) -> list[float]:
    n = len(values)
    first_season_avg = sum(values[:m]) / m
    level = first_season_avg
    trend = (sum(values[m : 2 * m]) / m - first_season_avg) / m

    if additive:
        seasons = [values[i] - first_season_avg for i in range(m)]
    else:
        seasons = [values[i] / first_season_avg if first_season_avg != 0 else 1.0 for i in range(m)]

    for t in range(m, n):
        v = values[t]
        s_idx = t % m
        prev_level = level
        if additive:
            level = alpha * (v - seasons[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasons[s_idx] = gamma * (v - level) + (1 - gamma) * seasons[s_idx]
        else:
            level = alpha * (v / seasons[s_idx] if seasons[s_idx] != 0 else v) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasons[s_idx] = gamma * (v / level if level != 0 else 1.0) + (1 - gamma) * seasons[s_idx]

    forecasts = []
    for step in range(1, h + 1):
        s_idx = (n - 1 + step) % m
        if additive:
            forecasts.append(level + step * trend + seasons[s_idx])
        else:
            forecasts.append((level + step * trend) * seasons[s_idx])
    return forecasts


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

try:
    from polars_ts_rs import ets_holt as _ets_holt_rs
    from polars_ts_rs import ets_holt_winters as _ets_hw_rs
    from polars_ts_rs import ets_ses as _ets_ses_rs

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ses_dispatch(values: list[float], alpha: float, h: int) -> list[float]:
    if _HAS_RUST:
        return _ets_ses_rs(values, alpha, h)
    return _ses_python(values, alpha, h)


def _holt_dispatch(values: list[float], alpha: float, beta: float, h: int) -> list[float]:
    if _HAS_RUST:
        return _ets_holt_rs(values, alpha, beta, h)
    return _holt_python(values, alpha, beta, h)


def _hw_dispatch(
    values: list[float], alpha: float, beta: float, gamma: float, m: int, additive: bool, h: int
) -> list[float]:
    if _HAS_RUST:
        return _ets_hw_rs(values, alpha, beta, gamma, m, additive, h)
    return _hw_python(values, alpha, beta, gamma, m, additive, h)


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------


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
        values = [float(v) for v in group_df[target_col].to_list()]
        last_time = group_df[time_col][-1]
        forecasts = _ses_dispatch(values, alpha, h)
        future_times = _make_future_dates(last_time, freq, h)
        for t, fc in zip(future_times, forecasts, strict=False):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": fc})

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
        values = [float(v) for v in group_df[target_col].to_list()]
        last_time = group_df[time_col][-1]

        if len(values) < 2:
            raise ValueError(f"Series {group_id[0]!r} needs at least 2 observations for Holt's method")

        forecasts = _holt_dispatch(values, alpha, beta, h)
        future_times = _make_future_dates(last_time, freq, h)
        for t, fc in zip(future_times, forecasts, strict=False):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": fc})

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

    additive = seasonal == "additive"
    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        values = [float(v) for v in group_df[target_col].to_list()]
        last_time = group_df[time_col][-1]
        m = season_length

        if len(values) < 2 * m:
            raise ValueError(
                f"Series {group_id[0]!r} needs at least 2*season_length={2 * m} observations, got {len(values)}"
            )

        forecasts = _hw_dispatch(values, alpha, beta, gamma, m, additive, h)
        future_times = _make_future_dates(last_time, freq, h)
        for t, fc in zip(future_times, forecasts, strict=False):
            rows.append({id_col: group_id[0], time_col: t, "y_hat": fc})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)
