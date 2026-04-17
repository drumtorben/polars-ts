"""Calendar feature extraction from datetime columns."""

from __future__ import annotations

import polars as pl

_EXTRACTORS: dict[str, object] = {
    "day_of_week": lambda c: c.dt.weekday().alias("day_of_week"),
    "day_of_month": lambda c: c.dt.day().alias("day_of_month"),
    "day_of_year": lambda c: c.dt.ordinal_day().alias("day_of_year"),
    "week": lambda c: c.dt.week().alias("week"),
    "month": lambda c: c.dt.month().alias("month"),
    "quarter": lambda c: c.dt.quarter().alias("quarter"),
    "year": lambda c: c.dt.year().alias("year"),
    "hour": lambda c: c.dt.hour().alias("hour"),
    "minute": lambda c: c.dt.minute().alias("minute"),
    "is_weekend": lambda c: (c.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
}


def calendar_features(
    df: pl.DataFrame,
    features: list[str] | None = None,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Extract calendar features from a datetime column.

    Parameters
    ----------
    df
        Input DataFrame.
    features
        List of calendar features to extract. Defaults to all available:
        ``day_of_week``, ``day_of_month``, ``day_of_year``, ``week``,
        ``month``, ``quarter``, ``year``, ``hour``, ``minute``,
        ``is_weekend``.
    time_col
        Datetime column to extract features from.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with calendar feature columns appended.

    """
    if features is None:
        features = list(_EXTRACTORS)

    for f in features:
        if f not in _EXTRACTORS:
            raise ValueError(f"Unknown calendar feature {f!r}. Choose from {sorted(_EXTRACTORS)}")

    col = pl.col(time_col)
    exprs = [_EXTRACTORS[f](col) for f in features]
    return df.with_columns(exprs)
