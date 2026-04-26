"""Gramian Angular Field (GAF) imaging for time series.

Converts time series to GASF (Gramian Angular Summation Field) and
GADF (Gramian Angular Difference Field) images.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_ts.imaging._utils import extract_series


def _min_max_scale(x: np.ndarray) -> np.ndarray:
    """Scale values to [-1, 1]."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return 2 * (x - xmin) / (xmax - xmin) - 1


def _paa(x: np.ndarray, size: int) -> np.ndarray:
    """Piecewise Aggregate Approximation — downsample by averaging segments."""
    n = len(x)
    if size >= n:
        return x
    indices = np.array_split(np.arange(n), size)
    return np.array([x[idx].mean() for idx in indices])


def _gasf_matrix(x: np.ndarray, image_size: int | None) -> np.ndarray:
    """Compute GASF for a single 1D series."""
    x_scaled = _min_max_scale(x)
    if image_size is not None:
        x_scaled = _paa(x_scaled, image_size)
    phi = np.arccos(np.clip(x_scaled, -1, 1))
    return np.cos(phi[:, None] + phi[None, :])


def _gadf_matrix(x: np.ndarray, image_size: int | None) -> np.ndarray:
    """Compute GADF for a single 1D series."""
    x_scaled = _min_max_scale(x)
    if image_size is not None:
        x_scaled = _paa(x_scaled, image_size)
    phi = np.arccos(np.clip(x_scaled, -1, 1))
    return np.sin(phi[:, None] - phi[None, :])


def to_gasf(
    df: pl.DataFrame,
    image_size: int | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to Gramian Angular Summation Field images.

    Rescales values to [-1, 1], converts to polar coordinates, and
    computes ``cos(phi_i + phi_j)`` for all pairs.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    image_size
        Output image dimension. ``None`` for full resolution (n x n).
        Smaller values use Piecewise Aggregate Approximation (PAA).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a square 2D array with values in [-1, 1].

    """
    series = extract_series(df, id_col, target_col)
    return {sid: _gasf_matrix(vals, image_size) for sid, vals in series.items()}


def to_gadf(
    df: pl.DataFrame,
    image_size: int | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to Gramian Angular Difference Field images.

    Rescales values to [-1, 1], converts to polar coordinates, and
    computes ``sin(phi_i - phi_j)`` for all pairs.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    image_size
        Output image dimension. ``None`` for full resolution (n x n).
        Smaller values use Piecewise Aggregate Approximation (PAA).
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a square 2D array with values in [-1, 1].

    """
    series = extract_series(df, id_col, target_col)
    return {sid: _gadf_matrix(vals, image_size) for sid, vals in series.items()}
