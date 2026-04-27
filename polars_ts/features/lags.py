"""Lag feature generation for time series data."""

from __future__ import annotations

import polars as pl


def lag_features(
    df: pl.DataFrame,
    lags: list[int],
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Create lagged versions of a target column per group.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    lags
        List of lag values (positive integers). Each produces a column
        ``{target_col}_lag_{k}``.
    target_col
        Column to lag.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with lag columns appended.

    """
    if any(k <= 0 for k in lags):
        raise ValueError("All lag values must be positive integers")

    sorted_df = df.sort(id_col, time_col)
    lag_exprs = [pl.col(target_col).shift(k).over(id_col).alias(f"{target_col}_lag_{k}") for k in lags]
    return sorted_df.with_columns(lag_exprs)


def covariate_lag_features(
    df: pl.DataFrame,
    covariate_cols: list[str],
    lags: list[int],
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Create lagged versions of covariate columns per group.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    covariate_cols
        Columns to lag.
    lags
        List of lag values (positive integers). Each produces a column
        ``{col}_lag_{k}`` per covariate.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with covariate lag columns appended.

    """
    if any(k <= 0 for k in lags):
        raise ValueError("All lag values must be positive integers")

    sorted_df = df.sort(id_col, time_col)
    lag_exprs = [pl.col(col).shift(k).over(id_col).alias(f"{col}_lag_{k}") for col in covariate_cols for k in lags]
    return sorted_df.with_columns(lag_exprs)
