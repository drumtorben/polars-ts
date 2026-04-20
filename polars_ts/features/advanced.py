"""Advanced feature engineering: target encoding, holidays, interactions, time embeddings. Closes #53."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def target_encode(
    df: pl.DataFrame,
    cat_col: str,
    target_col: str = "y",
    smoothing: float = 10.0,
    _id_col: str = "unique_id",
) -> pl.DataFrame:
    """Encode a categorical column by smoothed per-category target mean.

    Uses regularized (smoothed) encoding:
    ``encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)``

    Parameters
    ----------
    df
        Input DataFrame.
    cat_col
        Categorical column to encode.
    target_col
        Target column for computing means.
    smoothing
        Smoothing factor (higher = more regularization).
    id_col
        Not used directly but kept for API consistency.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``{cat_col}_encoded`` column appended.

    """
    global_mean = df[target_col].mean()

    cat_stats = df.group_by(cat_col).agg(
        pl.col(target_col).mean().alias("__cat_mean"),
        pl.col(target_col).len().alias("__cat_n"),
    )
    cat_stats = cat_stats.with_columns(
        ((pl.col("__cat_n") * pl.col("__cat_mean") + smoothing * global_mean) / (pl.col("__cat_n") + smoothing)).alias(
            f"{cat_col}_encoded"
        )
    ).select(cat_col, f"{cat_col}_encoded")

    return df.join(cat_stats, on=cat_col, how="left")


def holiday_features(
    df: pl.DataFrame,
    country: str = "US",
    time_col: str = "ds",
    distance: bool = False,
) -> pl.DataFrame:
    """Add binary holiday columns and optional distance-to-holiday features.

    Requires the ``holidays`` package (``pip install holidays``).

    Parameters
    ----------
    df
        Input DataFrame.
    country
        ISO country code (e.g. ``"US"``, ``"DE"``, ``"BR"``).
    time_col
        Datetime or date column.
    distance
        If ``True``, add ``days_to_holiday`` and ``days_since_holiday`` columns.

    """
    try:
        import holidays as holidays_lib
    except ImportError:
        raise ImportError("The 'holidays' package is required. Install with: pip install holidays") from None

    dates = df[time_col].to_list()
    # Extract date objects
    date_objs = []
    for d in dates:
        if hasattr(d, "date"):
            date_objs.append(d.date())
        else:
            date_objs.append(d)

    years = sorted({d.year for d in date_objs})
    cal = holidays_lib.country_holidays(country, years=years)

    is_holiday = [d in cal for d in date_objs]
    result = df.with_columns(pl.Series("is_holiday", is_holiday).cast(pl.Int8))

    if distance:
        holiday_dates = sorted(cal.keys())
        days_to: list[int] = []
        days_since: list[int] = []
        for d in date_objs:
            future = [h for h in holiday_dates if h >= d]
            past = [h for h in holiday_dates if h <= d]
            days_to.append((future[0] - d).days if future else 365)
            days_since.append((d - past[-1]).days if past else 365)
        result = result.with_columns(
            pl.Series("days_to_holiday", days_to),
            pl.Series("days_since_holiday", days_since),
        )

    return result


def interaction_features(
    df: pl.DataFrame,
    pairs: list[tuple[str, str]],
    method: str = "multiply",
) -> pl.DataFrame:
    """Generate interaction features between pairs of columns.

    Parameters
    ----------
    df
        Input DataFrame.
    pairs
        List of ``(col_a, col_b)`` tuples.
    method
        ``"multiply"`` (default) or ``"add"``.

    """
    if method not in ("multiply", "add"):
        raise ValueError(f"method must be 'multiply' or 'add', got {method!r}")

    result = df
    for col_a, col_b in pairs:
        name = f"{col_a}_x_{col_b}" if method == "multiply" else f"{col_a}_plus_{col_b}"
        if method == "multiply":
            result = result.with_columns((pl.col(col_a) * pl.col(col_b)).alias(name))
        else:
            result = result.with_columns((pl.col(col_a) + pl.col(col_b)).alias(name))

    return result


def time_embeddings(
    df: pl.DataFrame,
    time_col: str = "ds",
    components: list[str] | None = None,
) -> pl.DataFrame:
    """Encode cyclical time features as sin/cos pairs.

    Parameters
    ----------
    df
        Input DataFrame.
    time_col
        Datetime column.
    components
        Time components to encode. Defaults to ``["hour", "day_of_week", "month"]``.
        Each produces a sin/cos pair.

    """
    if components is None:
        components = ["hour", "day_of_week", "month"]

    _periods: dict[str, tuple[Any, float]] = {
        "hour": (lambda c: c.dt.hour(), 24.0),
        "day_of_week": (lambda c: c.dt.weekday(), 7.0),
        "day_of_month": (lambda c: c.dt.day(), 31.0),
        "day_of_year": (lambda c: c.dt.ordinal_day(), 366.0),
        "week": (lambda c: c.dt.week(), 53.0),
        "month": (lambda c: c.dt.month(), 12.0),
        "quarter": (lambda c: c.dt.quarter(), 4.0),
    }

    result = df
    col = pl.col(time_col)
    for comp in components:
        if comp not in _periods:
            raise ValueError(f"Unknown component {comp!r}. Choose from {sorted(_periods)}")
        extractor, period = _periods[comp]
        val = extractor(col).cast(pl.Float64)
        angle = 2 * np.pi * val / period
        result = result.with_columns(
            angle.sin().alias(f"{comp}_sin"),
            angle.cos().alias(f"{comp}_cos"),
        )

    return result
