"""Forecast error metrics for time series evaluation.

Implements standard metrics from Ch 19 of "Modern Time Series Forecasting
with Python" (2nd Ed.): MAE, RMSE, MAPE, sMAPE, MASE, and CRPS.
"""

from __future__ import annotations

import polars as pl


def mae(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
) -> pl.DataFrame | float:
    """Mean Absolute Error.

    Parameters
    ----------
    df
        DataFrame with actual and predicted columns.
    actual_col
        Column name for actual values.
    predicted_col
        Column name for predicted values.
    id_col
        If provided, compute MAE per group and return a DataFrame
        with columns ``[id_col, "mae"]``. Otherwise return a single float.

    Returns
    -------
    pl.DataFrame | float

    """
    expr = (pl.col(actual_col) - pl.col(predicted_col)).abs().mean().alias("mae")
    if id_col is not None:
        return df.group_by(id_col).agg(expr).sort(id_col)
    return df.select(expr).item()


def rmse(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
) -> pl.DataFrame | float:
    """Root Mean Squared Error.

    Parameters
    ----------
    df
        DataFrame with actual and predicted columns.
    actual_col
        Column name for actual values.
    predicted_col
        Column name for predicted values.
    id_col
        If provided, compute RMSE per group.

    Returns
    -------
    pl.DataFrame | float

    """
    expr = ((pl.col(actual_col) - pl.col(predicted_col)) ** 2).mean().sqrt().alias("rmse")
    if id_col is not None:
        return df.group_by(id_col).agg(expr).sort(id_col)
    return df.select(expr).item()


def mape(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
) -> pl.DataFrame | float:
    """Mean Absolute Percentage Error.

    Undefined when actual values are zero. Rows where ``actual == 0`` are
    excluded from the computation.

    Parameters
    ----------
    df
        DataFrame with actual and predicted columns.
    actual_col
        Column name for actual values.
    predicted_col
        Column name for predicted values.
    id_col
        If provided, compute MAPE per group.

    Returns
    -------
    pl.DataFrame | float
        MAPE as a fraction (not percentage). Multiply by 100 for percent.

    """
    filtered = df.filter(pl.col(actual_col) != 0)
    expr = ((pl.col(actual_col) - pl.col(predicted_col)) / pl.col(actual_col)).abs().mean().alias("mape")
    if id_col is not None:
        return filtered.group_by(id_col).agg(expr).sort(id_col)
    return filtered.select(expr).item()


def smape(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str | None = None,
) -> pl.DataFrame | float:
    """Symmetric Mean Absolute Percentage Error.

    Uses the formula: ``mean(2 * |actual - predicted| / (|actual| + |predicted|))``.
    Rows where both actual and predicted are zero are excluded.

    Parameters
    ----------
    df
        DataFrame with actual and predicted columns.
    actual_col
        Column name for actual values.
    predicted_col
        Column name for predicted values.
    id_col
        If provided, compute sMAPE per group.

    Returns
    -------
    pl.DataFrame | float
        sMAPE as a fraction (0 to 2). Multiply by 100 for the 0–200 scale.

    """
    denom = pl.col(actual_col).abs() + pl.col(predicted_col).abs()
    filtered = df.filter(denom != 0)
    expr = (2 * (pl.col(actual_col) - pl.col(predicted_col)).abs() / denom).mean().alias("smape")
    if id_col is not None:
        return filtered.group_by(id_col).agg(expr).sort(id_col)
    return filtered.select(expr).item()


def mase(
    df: pl.DataFrame,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
    season_length: int = 1,
) -> pl.DataFrame | float:
    """Mean Absolute Scaled Error.

    Scales the MAE by the in-sample naive forecast error. A MASE < 1 means
    the model outperforms the naive (seasonal) baseline.

    Parameters
    ----------
    df
        DataFrame with actual, predicted, and time columns.
    actual_col
        Column name for actual values.
    predicted_col
        Column name for predicted values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    season_length
        Seasonal period for the naive baseline. Use 1 for non-seasonal data
        (equivalent to random walk baseline).

    Returns
    -------
    pl.DataFrame | float
        If multiple series exist (via ``id_col``), returns a DataFrame with
        columns ``[id_col, "mase"]``. If only one series, returns a float.

    """
    # Sort by time within each group
    sorted_df = df.sort(id_col, time_col)

    # Compute per-group naive MAE (seasonal lag difference)
    naive_mae = (
        sorted_df.with_columns(
            (pl.col(actual_col) - pl.col(actual_col).shift(season_length).over(id_col)).abs().alias("__naive_err")
        )
        .group_by(id_col)
        .agg(pl.col("__naive_err").mean().alias("__naive_mae"))
    )

    # Compute per-group forecast MAE
    forecast_mae = sorted_df.group_by(id_col).agg(
        (pl.col(actual_col) - pl.col(predicted_col)).abs().mean().alias("__forecast_mae")
    )

    # Join and compute MASE (0/0 = 0 when both forecast and naive are perfect)
    result = forecast_mae.join(naive_mae, on=id_col).with_columns(
        pl.when(pl.col("__naive_mae") == 0)
        .then(pl.when(pl.col("__forecast_mae") == 0).then(0.0).otherwise(float("inf")))
        .otherwise(pl.col("__forecast_mae") / pl.col("__naive_mae"))
        .alias("mase")
    )

    ids = result[id_col].to_list()
    if len(ids) == 1:
        return result["mase"].item()

    return result.select(id_col, "mase").sort(id_col)


def crps(
    df: pl.DataFrame,
    actual_col: str = "y",
    quantile_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
    id_col: str | None = None,
) -> pl.DataFrame | float:
    """Continuous Ranked Probability Score (quantile approximation).

    Approximates CRPS using a set of quantile forecasts via the pinball
    (quantile) loss, averaged across quantiles.

    Parameters
    ----------
    df
        DataFrame with actual values and quantile forecast columns.
    actual_col
        Column name for actual values.
    quantile_cols
        List of column names containing quantile forecasts. If ``None``,
        auto-detected as columns matching ``q_*`` pattern.
    quantiles
        List of quantile levels (e.g. ``[0.1, 0.5, 0.9]``) corresponding
        to ``quantile_cols``. If ``None``, parsed from column names
        (e.g. ``"q_0.1"`` → ``0.1``).
    id_col
        If provided, compute CRPS per group.

    Returns
    -------
    pl.DataFrame | float

    """
    if quantile_cols is None:
        quantile_cols = [c for c in df.columns if c.startswith("q_")]
    if not quantile_cols:
        raise ValueError("No quantile columns found. Provide quantile_cols or use columns named 'q_0.1', 'q_0.5', etc.")

    if quantiles is None:
        try:
            quantiles = [float(c.split("_", 1)[1]) for c in quantile_cols]
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Cannot parse quantile levels from column names {quantile_cols}. "
                "Expected format 'q_0.1', 'q_0.5', etc., or provide explicit quantiles."
            ) from e

    if len(quantiles) != len(quantile_cols):
        raise ValueError(f"Length mismatch: {len(quantiles)} quantiles vs {len(quantile_cols)} columns")

    # Pinball loss for each quantile: q * max(y - y_q, 0) + (1-q) * max(y_q - y, 0)
    pinball_exprs = []
    for q, col in zip(quantiles, quantile_cols, strict=True):
        error = pl.col(actual_col) - pl.col(col)
        pinball = pl.when(error >= 0).then(q * error).otherwise((q - 1) * error)
        pinball_exprs.append(pinball.alias(f"__pinball_{col}"))

    with_pinball = df.with_columns(pinball_exprs)
    pinball_cols = [f"__pinball_{c}" for c in quantile_cols]

    # Average across quantiles, then across samples
    avg_expr = pl.concat_list(pinball_cols).list.mean().alias("__avg_pinball")
    with_avg = with_pinball.with_columns(avg_expr)

    crps_expr = pl.col("__avg_pinball").mean().alias("crps")
    if id_col is not None:
        return with_avg.group_by(id_col).agg(crps_expr).sort(id_col)
    return with_avg.select(crps_expr).item()
