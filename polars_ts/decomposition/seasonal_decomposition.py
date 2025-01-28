from typing import Literal

import polars as pl


def seasonal_decomposition(
    df: pl.DataFrame,
    freq: int,
    method: Literal["additive", "multiplicative"] = "additive",
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Perform seasonal decomposition of time series data using either an additive or multiplicative method.

    Additive: `Y(t) = T(t) + S(t) + E(t)`
    Multiplicative: `Y(t) = T(t) * S(t) * E(t)`

    Args:
        df: Polars DataFrame containing the time series data.
        freq: The seasonal period (e.g., 12 for monthly data with yearly seasonality).
        method: The decomposition method (additive or 'multiplicative').
        id_col: The column to group by (e.g., for multiple time series).
        target_col: The column containing the time series values to decompose.
        time_col: The column containing the time values.

    Returns:
        Polars DataFrame with the decomposed components: trend, seasonal component, and residuals.

    """
    period_idx = pl.col(time_col).cum_count().mod(freq).over(id_col).alias("period_idx")

    # Trend: Rolling mean with window size = freq
    trend_expr = pl.col(target_col).rolling_mean(window_size=freq, center=True).over(id_col).alias("trend")

    if method == "additive":
        func = pl.Expr.sub
    elif method == "multiplicative":
        func = pl.Expr.truediv

    # Seasonal component (additive method)
    seasonal_component_expr = (
        pl.col(target_col).pipe(func, "trend").mean().over(id_col, "period_idx").alias("seasonal_idx")
    )

    # Adjust seasonal component to have mean = 0 (for additive)
    seasonal_idx_expr = pl.col("seasonal_idx").sub(pl.col("seasonal_idx").mean().over(id_col)).alias(f"seasonal_{freq}")

    # Residuals:
    # Original series - trend - seasonal components (additive)
    # Original series / trend / seasonal components (multiplicative)
    residuals_expr = pl.col(target_col).pipe(func, pl.col("trend")).pipe(func, pl.col(f"seasonal_{freq}"))

    df = (
        df.with_columns(period_idx, trend_expr)
        .with_columns(seasonal_component_expr)
        .with_columns(seasonal_idx_expr)
        .with_columns(residuals_expr.alias("resid"))
        .drop("period_idx", "seasonal_idx")
        # drop nulls created by centered moving average
        .drop_nulls()
    )

    return df
