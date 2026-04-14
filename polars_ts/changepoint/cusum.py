import polars as pl


def cusum(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    normalize: bool = True,
) -> pl.DataFrame:
    """Compute the cumulative sum (CUSUM) of deviations from the mean.

    CUSUM detects shifts in the mean level of a time series. The statistic
    accumulates deviations from the overall mean:
    ``C[t] = sum_{i=1}^{t} (x[i] - mean(x))``.
    A sustained change in mean produces a clear slope change in the CUSUM curve.

    Args:
        df: Polars DataFrame containing the time series data.
        target_col: The column containing the time series values. Defaults to ``y``.
        id_col: The column to group by for multiple time series. Defaults to ``unique_id``.
        normalize: If True, divide by the standard deviation to get a standardized
            CUSUM (unitless). Defaults to True.

    Returns:
        The input DataFrame with an additional ``cusum`` column.

    Raises:
        ValueError: If the DataFrame is empty.
        KeyError: If the required columns are missing.

    """
    required_columns = {id_col, target_col}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Columns {sorted(missing)} are missing from the DataFrame.")

    if df.is_empty():
        raise ValueError("The DataFrame is empty. Cannot compute CUSUM on an empty DataFrame.")

    result = df.with_columns(
        pl.col(target_col).sub(pl.col(target_col).mean().over(id_col)).alias("__cusum_dev")
    )

    if normalize:
        result = result.with_columns(
            pl.col(target_col).std().over(id_col).alias("__cusum_std")
        )
        result = result.with_columns(
            pl.when(pl.col("__cusum_std") > 0.0)
            .then(pl.col("__cusum_dev") / pl.col("__cusum_std"))
            .otherwise(0.0)
            .cum_sum()
            .over(id_col)
            .alias("cusum")
        ).drop("__cusum_dev", "__cusum_std")
    else:
        result = result.with_columns(
            pl.col("__cusum_dev").cum_sum().over(id_col).alias("cusum")
        ).drop("__cusum_dev")

    return result
