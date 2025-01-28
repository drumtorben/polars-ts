from typing import List, Literal

import polars as pl
import polars_ds as pds


def fourier_decomposition(
    df: pl.DataFrame,
    ts_freq: int,
    freqs: List[Literal["week", "month", "quarter", "day_of_week", "day_of_month", "day_of_year"]] = ["week"],
    n_fourier_terms: int = 3,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Perform Fourier decomposition on a time series dataset to extract trend, seasonal, and residual components.
    The decomposition is based on Fourier harmonics for various temporal frequencies (e.g., week, month, quarter).

    Parameters
    ----------
    df : pl.DataFrame
        The input Polars DataFrame containing the time series data.

    id_col : str
        The name of the column that uniquely identifies each row (e.g., "id").

    time_col : str
        The name of the column containing the timestamps or time values. This is used to generate temporal features like "week", "month", etc.

    target_col : str
        The name of the target variable (column) whose seasonal and trend components are being decomposed.

    ts_freq: int
        the number of periods within a seasonal cycle: 52 for weekly data, 4 for quarterly data, 12 for monthly data, etc.....

    freqs : List[Literal['week', 'month', 'quarter', 'day_of_week', 'day_of_month', 'day_of_year']], optional (default is ['week'])
        A list of frequencies to use for generating Fourier harmonics. Options include:
        - 'week' (weekly frequency)
        - 'month' (monthly frequency)
        - 'quarter' (quarterly frequency)
        - 'day_of_week' (day of the week, 0-6)
        - 'day_of_month' (day of the month, 1-31)
        - 'day_of_year' (day of the year, 1-365/366)

    n_fourier_terms : int, optional (default is 3)
        The number of Fourier terms (harmonics) to generate for each frequency. Higher values allow capturing more complex seasonal patterns.

    Returns
    -------
    pl.DataFrame
        A DataFrame with the following columns:
        - `id_col`: The original ID column.
        - `time_col`: The original time column.
        - `target_col`: The original target variable.
        - `trend`: The estimated trend component (using moving average).
        - `seasonal`: The seasonal component (estimated using Fourier harmonics).
        - `resid`: The residuals, computed as the difference between the original target and the sum of the trend and seasonal components.

    """
    freq_dict = {
        "week": pl.col(time_col).dt.week().alias("week"),
        "month": pl.col(time_col).dt.month().alias("month"),
        "quarter": pl.col(time_col).dt.quarter().alias("quarter"),
        "day_of_week": pl.col(time_col).dt.weekday().alias("day_of_week"),
        "day_of_month": pl.col(time_col).dt.day().alias("day_of_month"),
        "day_of_year": pl.col(time_col).dt.ordinal_day().alias("day_of_year"),
    }

    # Trend: Rolling mean with window size = freq
    trend_expr = pl.col(target_col).rolling_mean(window_size=ts_freq, center=True).over(id_col).alias("trend")

    # generate date features for all keys in freq dict
    date_features = [freq_dict[freq] for freq in freqs]

    # generate harmonic pairs
    generate_harmonics = [
        [pl.col(freq).mul(i).sin().over(id_col).name.suffix(f"_sin_{i}") for freq in freqs]
        + [pl.col(freq).mul(i).cos().over(id_col).name.suffix(f"_cos_{i}") for freq in freqs]
        for i in range(1, n_fourier_terms + 1)
    ]

    # flatten the nested lists into a single list of expressions
    harmonic_expr = [pair for sublist in generate_harmonics for pair in sublist]

    # add date features and harmonics
    df = df.with_columns(*date_features).with_columns(*harmonic_expr)

    # these are all the sine/cosine pairs in the data, pds doesn't play nice with polars.selectors :(
    independent_vars = [col for col in df.columns if "_cos" in col or "_sin" in col]

    # detrend the series using Moving Averages, fit linear regression with fourier terms as features.
    result = (
        df.with_columns(trend_expr)
        .drop_nulls()  # drop nulls created by moving average
        .with_columns(pl.col(target_col).sub(pl.col("trend")).over(id_col).alias(f"{target_col}_detrend"))
        # fit linear regression on detrended data
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
