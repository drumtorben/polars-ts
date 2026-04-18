"""Fourier (sin/cos harmonic) feature generation for time series data."""

from __future__ import annotations

import math

import polars as pl


def fourier_features(
    df: pl.DataFrame,
    period: float,
    n_harmonics: int = 1,
    time_col: str = "ds",
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Generate Fourier sin/cos pairs for seasonal modelling.

    Creates ``2 * n_harmonics`` columns using a within-group time index
    (0, 1, 2, ...) to construct harmonics of the given period.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    period
        Seasonal period length (e.g. 7 for weekly, 365.25 for yearly).
    n_harmonics
        Number of harmonic pairs (sin + cos) to generate.
    time_col
        Column with timestamps for ordering.
    id_col
        Column identifying each time series.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with Fourier feature columns appended.

    """
    if period <= 0:
        raise ValueError("period must be positive")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")

    period = float(period)
    sorted_df = df.sort(id_col, time_col)

    # Create a within-group time index (0-based row number)
    t = pl.int_range(pl.len()).over(id_col).cast(pl.Float64)

    exprs: list[pl.Expr] = []
    for k in range(1, n_harmonics + 1):
        angle = 2 * math.pi * k * t / period
        exprs.append(angle.sin().alias(f"fourier_sin_{period}_{k}"))
        exprs.append(angle.cos().alias(f"fourier_cos_{period}_{k}"))

    return sorted_df.with_columns(exprs)
