"""Log transform and inverse for time series target columns."""

from __future__ import annotations

import polars as pl


def log_transform(
    df: pl.DataFrame,
    target_col: str = "y",
) -> pl.DataFrame:
    """Replace target column with log1p(target_col).

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to transform.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` replaced by its log1p and
        ``{target_col}_original`` preserving pre-transform values.

    """
    orig_col = f"{target_col}_original"
    if orig_col in df.columns:
        raise ValueError(f"Column {orig_col!r} already exists — transform may have been applied already")

    min_val = df[target_col].min()
    if min_val is not None and min_val <= -1:
        raise ValueError(f"log1p requires values > -1, found {min_val}")

    return df.with_columns(
        pl.col(target_col).alias(orig_col),
        pl.col(target_col).log1p(),
    )


def inverse_log_transform(
    df: pl.DataFrame,
    target_col: str = "y",
) -> pl.DataFrame:
    """Replace target column with expm1(target_col), restoring original scale.

    Parameters
    ----------
    df
        DataFrame with a log-transformed target column.
    target_col
        Column to invert.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` restored and metadata columns dropped.

    """
    result = df.with_columns(
        (pl.col(target_col).exp() - 1).alias(target_col),
    )

    orig_col = f"{target_col}_original"
    if orig_col in result.columns:
        result = result.drop(orig_col)

    return result
