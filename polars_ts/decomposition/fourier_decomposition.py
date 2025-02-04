from typing import Literal, Tuple

import polars as pl
import polars_ds as pds


def fourier_decomposition(
    df: pl.DataFrame,
    ts_freq: int,
    freqs: Tuple[Literal["week", "month", "quarter", "day_of_week", "day_of_month", "day_of_year"]] = ("week",),
    n_fourier_terms: int = 3,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Perform Fourier decomposition on a time series dataset.

    Extract trend, seasonal, and residual components.
    The decomposition is based on Fourier harmonics for
    various temporal frequencies (e.g., week, month, quarter).

    Args:
        df: The input Polars DataFrame containing the time series data.
        ts_freq: The number of periods within a seasonal cycle: 52 for weekly data,
            4 for quarterly data, 12 for monthly data, etc.
        freqs : A tuple of frequencies to use for generating Fourier harmonics. Options include:

            - `week` (weekly frequency)
            - `month` (monthly frequency)
            - `quarter` (quarterly frequency)
            - `day_of_week` (day of the week, 0-6)
            - `day_of_month` (day of the month, 1-31)
            - `day_of_year` (day of the year, 1-365/366)
        n_fourier_terms: The number of Fourier terms (harmonics) to generate for each frequency.
            Higher values allow capturing more complex seasonal patterns.
        id_col: The name of the column that uniquely identifies each series (default: `unique_id`).
        time_col: The name of the column containing the timestamps or time values.
            This is used to generate temporal features like "week", "month", etc. Defaults to `ds`.
        target_col: The name of the target variable (column) whose seasonal and
            trend components are being decomposed. Defaults to `y`.

    Returns:
        A DataFrame with the following columns:

            - `id_col`: The original ID column.
            - `time_col`: The original time column.
            - `target_col`: The original target variable.
            - `trend`: The estimated trend component (using moving average).
            - `seasonal`: The seasonal component (estimated using Fourier harmonics).
            - `resid`: The residuals, computed as the difference between the original
                target and the sum of the trend and seasonal components.

    """
    # Check if necessary columns exist in the dataframe
    required_columns = {id_col, target_col, time_col}

    assert required_columns.issubset(df.columns), KeyError(
        f"Columns {required_columns.difference(df.columns)} are missing from the DataFrame."
    )

    # Validate ts_freq: ensure it's a positive integer
    if not isinstance(ts_freq, int) or ts_freq <= 0:
        raise ValueError(f"Invalid ts_freq '{ts_freq}'. It must be a positive integer.")

    # Validate freqs: ensure all frequencies are valid
    valid_freqs = ["week", "month", "quarter", "day_of_week", "day_of_month", "day_of_year"]

    assert set(freqs).issubset(valid_freqs), KeyError(
        f"Invalid Frequencies {set(freqs).difference(valid_freqs)}, \
            please pass any combination of elements in {valid_freqs}"
    )

    # Validate n_fourier_terms: ensure it's a positive integer
    if not isinstance(n_fourier_terms, int) or n_fourier_terms <= 0:
        raise ValueError(f"Invalid n_fourier_terms '{n_fourier_terms}'. It must be a positive integer.")

    # Ensure the dataframe is not empty
    if df.shape[0] == 0:
        raise ValueError("The DataFrame is empty. Cannot perform decomposition on an empty DataFrame.")

    # define expression list...
    expr_list = [
        pl.col(time_col).dt.week().alias("week"),
        pl.col(time_col).dt.month().alias("month"),
        pl.col(time_col).dt.quarter().alias("quarter"),
        pl.col(time_col).dt.weekday().alias("day_of_week"),
        pl.col(time_col).dt.day().alias("day_of_month"),
        pl.col(time_col).dt.ordinal_day().alias("day_of_year"),
    ]

    # Define frequency mapping for temporal features
    freq_dict = dict(zip(valid_freqs, expr_list, strict=False))

    # Trend: Rolling mean with window size = ts_freq
    trend_expr = pl.col(target_col).rolling_mean(window_size=ts_freq, center=True).over(id_col).alias("trend")

    # Generate date features for all keys in freq_dict
    date_features = [freq_dict[freq] for freq in freqs]

    # Generate harmonic pairs (sine and cosine components for each frequency)
    generate_harmonics = [
        [pl.col(freq).mul(i).sin().over(id_col).name.suffix(f"_sin_{i}") for freq in freqs]
        + [pl.col(freq).mul(i).cos().over(id_col).name.suffix(f"_cos_{i}") for freq in freqs]
        for i in range(1, n_fourier_terms + 1)
    ]

    # Flatten the nested list of harmonic expressions
    harmonic_expr = [pair for sublist in generate_harmonics for pair in sublist]

    # Add date features and harmonics to the dataframe
    df = df.with_columns(*date_features).with_columns(*harmonic_expr)

    # These are the sine/cosine pairs in the data
    independent_vars = [col for col in df.columns if "_cos" in col or "_sin" in col]

    # Detrend the series using Moving Averages, and fit linear regression with Fourier terms as features
    result = (
        df.with_columns(trend_expr)
        .drop_nulls()  # Drop nulls created by moving average
        .with_columns(pl.col(target_col).sub(pl.col("trend")).over(id_col).alias(f"{target_col}_detrend"))
        .with_columns(
            pds.lin_reg(*independent_vars, target=target_col + "_detrend", return_pred=True, l2_reg=0.001)
            .over(id_col)
            .struct.field("pred")
            .alias("seasonal")
        )
        .with_columns(pl.col("trend").add(pl.col("seasonal")).sub(pl.col(target_col)).over(id_col).alias("resid"))
        .select(id_col, time_col, target_col, "trend", "seasonal", "resid")
    )

    return result
