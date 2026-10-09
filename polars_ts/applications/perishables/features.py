"""Perishables-specific feature engineering.

Domain features layered on top of the generic helpers
(:func:`polars_ts.lag_features`, :func:`polars_ts.rolling_features`,
:func:`polars_ts.calendar_features`): retail calendar flags, per-SKU
day-of-week demand profiles (fresh-goods demand is dominated by weekly
shopping rhythm), and promotion lift estimation.
"""

from __future__ import annotations

import polars as pl

from polars_ts.features.calendar import calendar_features

_EPS = 1e-9


def perishable_calendar_features(
    df: pl.DataFrame,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Add retail calendar features relevant to perishables demand.

    Extends the generic calendar features (day of week, month, weekend
    flag) with month-boundary flags that proxy payday-driven grocery
    spikes: ``is_month_start`` (first 5 days) and ``is_month_end``
    (last 5 days).

    Parameters
    ----------
    df
        Canonical sales frame.
    time_col
        Date column.

    Returns
    -------
    pl.DataFrame
        Input frame with ``day_of_week``, ``day_of_month``, ``week``,
        ``month``, ``is_weekend``, ``is_month_start``, ``is_month_end``
        appended.

    """
    out = calendar_features(df, ["day_of_week", "day_of_month", "week", "month", "is_weekend"], time_col)
    days_in_month = pl.col(time_col).dt.month_end().dt.day()
    return out.with_columns(
        (pl.col("day_of_month") <= 5).cast(pl.Int8).alias("is_month_start"),
        (pl.col("day_of_month") > days_in_month - 5).cast(pl.Int8).alias("is_month_end"),
    )


def dow_profile(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Compute per-SKU day-of-week demand indices.

    The index is the SKU's mean demand on that weekday divided by its
    overall mean, so 1.0 is a neutral day and e.g. 1.8 on Saturday means
    an 80% uplift. Use it to shape flat intermittent-demand forecasts
    across the week before feeding the restock calculator.

    Parameters
    ----------
    df
        Gap-free canonical sales frame.
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "day_of_week", "dow_index"]`` (weekday 1 =
        Monday ... 7 = Sunday, matching polars).

    """
    return (
        df.with_columns(pl.col(time_col).dt.weekday().alias("day_of_week"))
        .group_by(id_col, "day_of_week")
        .agg(pl.col(target_col).mean().alias("__dow_mean"))
        .with_columns((pl.col("__dow_mean") / (pl.col("__dow_mean").mean().over(id_col) + _EPS)).alias("dow_index"))
        .drop("__dow_mean")
        .sort(id_col, "day_of_week")
    )


def estimate_promo_lift(
    df: pl.DataFrame,
    promo_col: str = "promo",
    target_col: str = "y",
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Estimate each SKU's multiplicative promotion lift.

    Lift is mean demand on promoted periods divided by mean demand on
    non-promoted periods. SKUs never (or always) promoted get a null
    lift — there is no counterfactual to compare against.

    Parameters
    ----------
    df
        Canonical sales frame with a promotion flag column.
    promo_col
        Column with a truthy promotion indicator.
    target_col, id_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, "promo_lift", "n_promo_periods"]``.

    """
    if promo_col not in df.columns:
        raise ValueError(f"Column {promo_col!r} not found; map it via ColumnMapping(promo=...)")
    on_promo = pl.col(promo_col).cast(pl.Boolean)
    return (
        df.group_by(id_col)
        .agg(
            (pl.col(target_col).filter(on_promo).mean() / (pl.col(target_col).filter(~on_promo).mean() + _EPS)).alias(
                "promo_lift"
            ),
            on_promo.sum().alias("n_promo_periods"),
        )
        .sort(id_col)
    )
