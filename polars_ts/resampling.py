"""Group-aware temporal resampling for time series. Closes #62."""

from __future__ import annotations

import polars as pl


def resample(
    df: pl.DataFrame,
    rule: str,
    agg: str = "mean",
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    fill: str | None = None,
) -> pl.DataFrame:
    """Resample a time series DataFrame to a new frequency.

    Parameters
    ----------
    df
        Input DataFrame.
    rule
        Target frequency as a Polars duration string (e.g. ``"1d"``,
        ``"1h"``, ``"1w"``).
    agg
        Aggregation for downsampling: ``"mean"``, ``"sum"``, ``"last"``,
        ``"first"``, ``"min"``, ``"max"``, ``"median"``.
    target_col
        Column to resample.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    fill
        Fill strategy for upsampling gaps: ``"forward_fill"``,
        ``"interpolate"``, or ``None`` (leave nulls).

    Returns
    -------
    pl.DataFrame
        Resampled DataFrame with columns ``[id_col, time_col, target_col]``.

    """
    valid_aggs = {"mean", "sum", "last", "first", "min", "max", "median"}
    if agg not in valid_aggs:
        raise ValueError(f"Unknown agg {agg!r}. Choose from {sorted(valid_aggs)}")

    agg_map = {
        "mean": pl.col(target_col).mean(),
        "sum": pl.col(target_col).sum(),
        "last": pl.col(target_col).last(),
        "first": pl.col(target_col).first(),
        "min": pl.col(target_col).min(),
        "max": pl.col(target_col).max(),
        "median": pl.col(target_col).median(),
    }

    sorted_df = df.sort(id_col, time_col)

    # Ensure time_col is Datetime for group_by_dynamic
    if sorted_df[time_col].dtype == pl.Date:
        sorted_df = sorted_df.with_columns(pl.col(time_col).cast(pl.Datetime("ms")))

    result = sorted_df.group_by_dynamic(
        time_col,
        every=rule,
        group_by=id_col,
    ).agg(agg_map[agg].alias(target_col))

    if fill == "forward_fill":
        result = result.with_columns(pl.col(target_col).forward_fill().over(id_col))
    elif fill == "interpolate":
        result = result.with_columns(pl.col(target_col).interpolate().over(id_col))

    return result.sort(id_col, time_col)
