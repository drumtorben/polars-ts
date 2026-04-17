"""Box-Cox power transform and inverse for time series target columns."""

from __future__ import annotations

import polars as pl


def boxcox_transform(
    df: pl.DataFrame,
    lam: float,
    target_col: str = "y",
) -> pl.DataFrame:
    """Apply Box-Cox power transform to target column in-place.

    Transform definition:

    - ``lambda == 0``: ``log(y)``
    - ``lambda != 0``: ``(y^lambda - 1) / lambda``

    Parameters
    ----------
    df
        Input DataFrame.
    lam
        Box-Cox lambda parameter.
    target_col
        Column to transform.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` replaced by its Box-Cox transform,
        ``{target_col}_original`` preserving pre-transform values, and
        ``{target_col}_boxcox_lambda`` storing the lambda for inversion.

    """
    orig_col = f"{target_col}_original"
    if orig_col in df.columns:
        raise ValueError(f"Column {orig_col!r} already exists — transform may have been applied already")

    min_val = df[target_col].min()
    if min_val is not None and min_val <= 0:
        raise ValueError(f"Box-Cox requires strictly positive values, found {min_val}")

    if lam == 0:
        expr = pl.col(target_col).log()
    else:
        expr = (pl.col(target_col).pow(lam) - 1) / lam

    return df.with_columns(
        pl.col(target_col).alias(orig_col),
        expr.alias(target_col),
        pl.lit(lam).alias(f"{target_col}_boxcox_lambda"),
    )


def inverse_boxcox_transform(
    df: pl.DataFrame,
    lam: float | None = None,
    target_col: str = "y",
) -> pl.DataFrame:
    """Invert Box-Cox transform on target column.

    Parameters
    ----------
    df
        DataFrame with a Box-Cox-transformed target column.
    lam
        Box-Cox lambda. If ``None``, read from
        ``{target_col}_boxcox_lambda`` column.
    target_col
        Column to invert.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` restored and metadata columns dropped.

    """
    lambda_col = f"{target_col}_boxcox_lambda"
    if lam is None:
        if lambda_col not in df.columns:
            raise ValueError(f"lam not provided and column {lambda_col!r} not found")
        lam = df[lambda_col][0]

    if lam == 0:
        expr = pl.col(target_col).exp()
    else:
        expr = (pl.col(target_col) * lam + 1).pow(1.0 / lam)

    result = df.with_columns(expr.alias(target_col))

    # Drop metadata columns
    to_drop = [c for c in [f"{target_col}_original", lambda_col] if c in result.columns]
    if to_drop:
        result = result.drop(to_drop)

    return result
