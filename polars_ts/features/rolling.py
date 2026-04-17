"""Rolling window feature generation for time series data."""

from __future__ import annotations

import polars as pl

_DEFAULT_AGGS = ["mean", "std", "min", "max"]

_SUPPORTED_AGGS = {"mean", "std", "min", "max", "sum", "median", "var"}


def rolling_features(
    df: pl.DataFrame,
    windows: list[int],
    aggs: list[str] | None = None,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    center: bool = False,
    min_samples: int | None = None,
) -> pl.DataFrame:
    """Create rolling window features for a target column per group.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    windows
        List of window sizes.
    aggs
        Aggregation functions to apply. Defaults to
        ``["mean", "std", "min", "max"]``. Supported: mean, std, min, max,
        sum, median, var.
    target_col
        Column to compute rolling features on.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    center
        Whether the rolling window is centred.
    min_samples
        Minimum number of non-null values required. Defaults to the window
        size when ``None``.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with rolling feature columns appended.

    """
    if aggs is None:
        aggs = list(_DEFAULT_AGGS)

    for agg in aggs:
        if agg not in _SUPPORTED_AGGS:
            raise ValueError(f"Unsupported aggregation {agg!r}. Choose from {sorted(_SUPPORTED_AGGS)}")

    if any(w <= 0 for w in windows):
        raise ValueError("All window sizes must be positive integers")

    sorted_df = df.sort(id_col, time_col)

    exprs: list[pl.Expr] = []
    for w in windows:
        mp = min_samples if min_samples is not None else w
        for agg in aggs:
            rolling_fn = getattr(pl.col(target_col), f"rolling_{agg}")
            expr = rolling_fn(w, min_samples=mp, center=center).over(id_col)
            exprs.append(expr.alias(f"{target_col}_rolling_{agg}_{w}"))

    return sorted_df.with_columns(exprs)
