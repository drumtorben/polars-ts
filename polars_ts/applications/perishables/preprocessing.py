"""Growth-aware preprocessing for rapidly scaling perishables demand.

A double-digit YoY growth business breaks the stationarity assumption of
most models: two-year-old observations describe a much smaller company.
Three complementary levers are provided:

- **Growth detrending** — estimate a per-SKU exponential growth rate and
  rescale history to today's level (and re-apply it to forecasts).
- **Recency weighting** — exponential-decay sample weights for models
  that accept them.
- **Adaptive windowing** — per-SKU training-window length chosen so the
  level drift across the window stays below a tolerance; fast-growing
  SKUs train on less (but recent) history.
"""

from __future__ import annotations

import math

import polars as pl

_EPS = 1e-9


def fit_growth(
    df: pl.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    smooth_window: int = 28,
) -> pl.DataFrame:
    """Estimate a per-SKU daily exponential growth rate.

    Fits ordinary least squares of ``log(rolling_mean(y) + eps)`` against
    time (in days) per SKU. The rolling mean smooths over zero-demand
    days so intermittent SKUs get a level trend, not noise.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    id_col, time_col, target_col
        Canonical column names.
    smooth_window
        Rolling-mean window (days) applied before the log-linear fit.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "daily_growth"]`` where ``daily_growth`` is
        the fitted slope of the log-level per day (0.0 for SKUs with
        fewer than ``smooth_window`` observations).

    """
    smoothed = df.sort(id_col, time_col).with_columns(
        pl.col(target_col).rolling_mean(window_size=smooth_window).over(id_col).alias("__level"),
        ((pl.col(time_col) - pl.col(time_col).min().over(id_col)).dt.total_days()).alias("__t"),
    )
    smoothed = smoothed.drop_nulls("__level").with_columns((pl.col("__level") + _EPS).log().alias("__log_level"))
    fitted = (
        smoothed.group_by(id_col)
        .agg(
            (pl.cov(pl.col("__t"), pl.col("__log_level")) / (pl.col("__t").var() + _EPS))
            .fill_null(0.0)
            .alias("daily_growth"),
            pl.len().alias("__n"),
        )
        .with_columns(pl.when(pl.col("__n") < 2).then(0.0).otherwise(pl.col("daily_growth")).alias("daily_growth"))
        .drop("__n")
    )
    all_ids = df.select(pl.col(id_col).unique())
    return all_ids.join(fitted, on=id_col, how="left").with_columns(pl.col("daily_growth").fill_null(0.0)).sort(id_col)


def remove_growth(
    df: pl.DataFrame,
    growth: pl.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Rescale history to today's demand level.

    Each observation is multiplied by ``exp(g * (t_last - t))`` so that a
    value observed a year ago on a SKU growing at rate *g* is expressed
    at the level the SKU sells at now. Models then see an (approximately)
    level-stationary series.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    growth
        Output of :func:`fit_growth`.
    id_col, time_col, target_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Input frame with ``target_col`` replaced by its level-adjusted
        value and the original preserved as ``"{target_col}_raw"``.

    """
    out = df.join(growth, on=id_col, how="left").with_columns(pl.col("daily_growth").fill_null(0.0))
    age = (pl.col(time_col).max().over(id_col) - pl.col(time_col)).dt.total_days()
    return (
        out.with_columns(pl.col(target_col).alias(f"{target_col}_raw"))
        .with_columns((pl.col(target_col) * (pl.col("daily_growth") * age).exp()).alias(target_col))
        .drop("daily_growth")
    )


def reapply_growth(
    forecast: pl.DataFrame,
    growth: pl.DataFrame,
    last_date: pl.DataFrame | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y_hat",
) -> pl.DataFrame:
    """Project the fitted growth back onto level-space forecasts.

    Parameters
    ----------
    forecast
        Forecast frame with ``[id_col, time_col, target_col]`` produced
        from growth-removed history.
    growth
        Output of :func:`fit_growth`.
    last_date
        Optional ``[id_col, time_col]`` frame with each SKU's last
        training date (defaults to one day before the first forecast
        date per SKU, which is correct for gap-free daily data).
    id_col, time_col, target_col
        Column names.

    Returns
    -------
    pl.DataFrame
        Forecast frame with ``target_col`` scaled by
        ``exp(g * days_ahead)``.

    """
    if last_date is None:
        last_date = forecast.group_by(id_col).agg((pl.col(time_col).min() - pl.duration(days=1)).alias("__last_train"))
    else:
        last_date = last_date.rename({time_col: "__last_train"})
    out = forecast.join(growth, on=id_col, how="left").join(last_date, on=id_col, how="left")
    days_ahead = (pl.col(time_col) - pl.col("__last_train")).dt.total_days()
    return (
        out.with_columns(pl.col("daily_growth").fill_null(0.0))
        .with_columns((pl.col(target_col) * (pl.col("daily_growth") * days_ahead).exp()).alias(target_col))
        .drop("daily_growth", "__last_train")
    )


def recency_weights(
    df: pl.DataFrame,
    half_life: float = 90.0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Add exponential-decay sample weights favouring recent history.

    Parameters
    ----------
    df
        Canonical sales frame.
    half_life
        Age in days at which a sample's weight halves.
    id_col, time_col
        Canonical column names.
    weight_col
        Name of the added weight column.

    Returns
    -------
    pl.DataFrame
        Input frame with ``weight_col`` in ``(0, 1]`` (1 at each SKU's
        most recent date).

    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    age = (pl.col(time_col).max().over(id_col) - pl.col(time_col)).dt.total_days()
    return df.with_columns((0.5 ** (age / half_life)).alias(weight_col))


def select_adaptive_window(
    growth: pl.DataFrame,
    tolerance: float = 0.25,
    min_days: int = 56,
    max_days: int = 730,
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Choose a per-SKU training-window length from its growth rate.

    The window is the longest span over which the fitted exponential
    level drifts by at most ``tolerance`` (25% by default):
    ``window = ln(1 + tolerance) / |daily_growth|``, clamped to
    ``[min_days, max_days]``. Stable SKUs keep long histories; SKUs
    doubling every few months train on short, recent windows.

    Parameters
    ----------
    growth
        Output of :func:`fit_growth`.
    tolerance
        Maximum tolerated relative level drift across the window.
    min_days, max_days
        Window clamp bounds in days.
    id_col
        SKU column name.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "window_days"]``.

    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if min_days <= 0 or max_days < min_days:
        raise ValueError("require 0 < min_days <= max_days")
    limit = math.log(1.0 + tolerance)
    return growth.select(
        pl.col(id_col),
        (limit / pl.col("daily_growth").abs().clip(lower_bound=_EPS))
        .clip(lower_bound=min_days, upper_bound=max_days)
        .cast(pl.Int64)
        .alias("window_days"),
    ).sort(id_col)


def apply_training_window(
    df: pl.DataFrame,
    windows: pl.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Trim each SKU's history to its adaptive training window.

    Parameters
    ----------
    df
        Canonical sales frame.
    windows
        Output of :func:`select_adaptive_window`.
    id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Rows within ``window_days`` of each SKU's last date; SKUs
        absent from ``windows`` are kept in full.

    """
    out = df.join(windows, on=id_col, how="left")
    age = (pl.col(time_col).max().over(id_col) - pl.col(time_col)).dt.total_days()
    return out.filter(pl.col("window_days").is_null() | (age < pl.col("window_days"))).drop("window_days")
