"""Forecast calibration diagnostics for probabilistic forecasts. Closes #58."""

from __future__ import annotations

import numpy as np
import polars as pl


def calibration_table(
    df: pl.DataFrame,
    actual_col: str = "y",
    quantile_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
    id_col: str | None = None,
) -> pl.DataFrame:
    """Compute observed vs expected coverage per quantile.

    Parameters
    ----------
    df
        DataFrame with actuals and quantile forecast columns.
    actual_col
        Column with actual values.
    quantile_cols
        Quantile forecast column names. Auto-detected from ``q_*`` if ``None``.
    quantiles
        Quantile levels. Parsed from column names if ``None``.
    id_col
        If provided, compute per group.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["quantile", "expected_coverage", "observed_coverage"]``.

    """
    quantile_cols, quantiles = _resolve_quantile_cols(df, quantile_cols, quantiles)

    rows: list[dict[str, float]] = []
    for q, col in zip(quantiles, quantile_cols, strict=False):
        # Observed coverage: fraction of actuals <= quantile forecast
        if id_col is not None:
            covered = df.group_by(id_col).agg((pl.col(actual_col) <= pl.col(col)).mean().alias("__cov"))["__cov"].mean()
        else:
            covered = float((df[actual_col] <= df[col]).mean())  # type: ignore[arg-type]
        rows.append({"quantile": q, "expected_coverage": q, "observed_coverage": float(covered)})  # type: ignore[arg-type]

    return pl.DataFrame(rows)


def pit_histogram(
    df: pl.DataFrame,
    actual_col: str = "y",
    quantile_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Compute Probability Integral Transform histogram data.

    A well-calibrated model produces a uniform PIT histogram.

    Parameters
    ----------
    df
        DataFrame with actuals and quantile forecast columns.
    actual_col
        Column with actual values.
    quantile_cols
        Quantile forecast column names.
    quantiles
        Quantile levels.
    n_bins
        Number of histogram bins.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["bin_lower", "bin_upper", "count", "density"]``.

    """
    quantile_cols, quantiles = _resolve_quantile_cols(df, quantile_cols, quantiles)

    # For each row, find the PIT value: fraction of quantile forecasts <= actual
    actuals = df[actual_col].to_numpy().astype(np.float64)
    q_matrix = df.select(quantile_cols).to_numpy().astype(np.float64)

    pit_values = np.mean(q_matrix <= actuals[:, None], axis=1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(pit_values, bins=bin_edges)
    total = len(pit_values)

    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        rows.append(
            {
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "count": float(counts[i]),
                "density": float(counts[i]) / total if total > 0 else 0.0,
            }
        )

    return pl.DataFrame(rows)


def reliability_diagram(
    df: pl.DataFrame,
    actual_col: str = "y",
    quantile_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> pl.DataFrame:
    """Compute data for a reliability (calibration) plot.

    Parameters
    ----------
    df
        DataFrame with actuals and quantile forecast columns.
    actual_col
        Column with actual values.
    quantile_cols
        Quantile forecast column names.
    quantiles
        Quantile levels.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["expected", "observed"]`` for plotting.

    """
    cal = calibration_table(df, actual_col, quantile_cols, quantiles)
    return cal.select(
        pl.col("expected_coverage").alias("expected"),
        pl.col("observed_coverage").alias("observed"),
    )


def _resolve_quantile_cols(
    df: pl.DataFrame,
    quantile_cols: list[str] | None,
    quantiles: list[float] | None,
) -> tuple[list[str], list[float]]:
    if quantile_cols is None:
        quantile_cols = sorted(c for c in df.columns if c.startswith("q_"))
    if not quantile_cols:
        raise ValueError("No quantile columns found. Provide quantile_cols or use columns named 'q_0.1', etc.")
    if quantiles is None:
        quantiles = [float(c.split("_", 1)[1]) for c in quantile_cols]
    return quantile_cols, quantiles
