"""Kaboudan Metrics Module.

Provides the `Kaboudan` class for computing Kaboudan and modified Kaboudan metrics to evaluate
time series forecasting models using backtesting and block shuffling techniques.
"""

import random
from dataclasses import dataclass

import polars as pl
from statsforecast import StatsForecast
from utilsforecast import losses


@dataclass
class Kaboudan:
    """A class for computing the Kaboudan and modified Kaboudan metrics.

    It uses StatsForecast for backtesting and block shuffling operations
    to measure model performance under controlled perturbations.

    Attributes:
        sf: StatsForecast instance for model training and evaluation.
        backtesting_start: Fraction of the data used as the initial training set.
        n_folds: Number of backtesting folds (rolling-origin windows).
        block_size: Size of each block used during block-based shuffling.
        seed: Random seed for reproducible shuffling. Defaults to 42.
        id_col: Name of the column identifying each time series group. Defaults to `unique_id`.
        time_col: Name of the column representing the chronological axis. Defaults to `ds`.
        value_col: Name of the column representing the target variable. Defaults to `y`.
        modified: Whether to use the modified Kaboudan metric, which applies clipping to zero. Defaults to `True`.
        agg: Whether to average the metrics over all the individual time series or not. Defaults to `True`.

    """

    sf: StatsForecast
    backtesting_start: float
    n_folds: int
    block_size: int
    seed: int = 42
    id_col: str = "unique_id"
    time_col: str = "ds"
    value_col: str = "y"
    modified: bool = True
    agg: bool = False

    def block_shuffle_by_id(self, df: pl.DataFrame) -> pl.DataFrame:
        """Randomly shuffles rows in fixed-size blocks within each group identified by `id_col`.

        This method sorts the data by `id_col` and then by `time_col`. For each group:  

        1. A zero-based row index (`__row_in_group`) is assigned using `cum_count()`.
        2. The method determines the number of blocks (`num_blocks`) by dividing the number of
        rows in the first group by `self.block_size` and forcing at least one block.
        3. Each row is assigned a `__chunk_id` based on integer division of `__row_in_group` by `num_blocks`.
        4. The DataFrame is then partitioned by both `id_col` and `__chunk_id`, producing blocks.
        5. These blocks are randomly shuffled, concatenated, and finally re-sorted by `id_col`
        and `time_col` within each group.

        Args:
            df: A Polars DataFrame containing at least `id_col`, `time_col`, and `value_col`.

        Returns:
            A new DataFrame in which each group's rows are rearranged by randomly shuffling \
            the entire blocks. The shuffle is reproducible if a seed is set (`self.seed`).

        """
        num_blocks = max(df.filter(pl.col(self.id_col) == pl.first(self.id_col)).height // self.block_size, 1)
        dfs = (
            df.sort(self.id_col, self.time_col)
            .with_columns(
                # cum_count() over each group to get the row index within that group
                pl.col(self.value_col).cum_count().over(self.id_col).alias("__row_in_group")
            )
            .with_columns(
                # __chunk_id = row_in_group // block_size
                (pl.col("__row_in_group") // num_blocks).alias("__chunk_id"),
            )
            .partition_by(self.id_col, "__chunk_id")
        )
        if self.seed is not None:
            # set seed
            random.seed(self.seed)
        random.shuffle(dfs)
        df = (
            pl.concat(dfs)
            .drop("__row_in_group", "__chunk_id")
            .sort(self.id_col)
            .with_columns(pl.col(self.time_col).sort().over(self.id_col))
        )
        return df

    def split_in_blocks_by_id(self, df: pl.DataFrame) -> pl.DataFrame:
        """Split each group's time series into `n_folds` sequential blocks.

        First, the DataFrame is sorted `by id_col` and `time_col`. Then, for each group (identified
        by `id_col`), a zero-based row index is assigned in `row_index`. Finally, `block` is
        computed by scaling `row_index` by the ratio `(n_folds / group_size)` for that group,
        flooring the result, and shifting by 1 to make blocks range from 1 to `n_folds`.

        Args:
            df: A DataFrame containing columns matching `id_col`, `time_col`, and `value_col`.

        Returns:
            A new DataFrame with one additional `block` column.

        """
        df = (
            df.sort(self.id_col, self.time_col)
            # Assign a zero-based row index per group
            .with_columns((pl.col(self.value_col).cum_count() - 1).over(self.id_col).alias("__row_index"))
            # Convert row_index to block assignments [1..n_folds]
            # block = floor(row_index * n_folds / group_size) + 1
            # where group_size is (max row_index in the group + 1).
            .with_columns(
                (((pl.col("__row_index") * self.n_folds) / pl.len()).over(self.id_col).floor() + 1)
                .cast(pl.Int32)
                .alias("block")
            )
            .drop("__row_index")
        )
        return df

    def backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        """Perform rolling-origin backtesting on the provided DataFrame using cross-validation.

        This method implements a multi-step cross-validation approach by:

        1. Computing the minimal series length among all groups in the DataFrame.
        2. Determining the initial training length (`history_len`) as `backtesting_start * min_len`,
        and setting the test length (`test_len`) as the remainder.
        3. Dividing the test portion into `n_folds` sequential segments. Each segment length
        determines the forecast horizon (`h`) and `step_size`.
        4. Calling StatsForecast's `cross_validation()` method with `h` and `step_size` both equal to
        the segment length.

        Args:
            df: A Polars DataFrame that must contain at least the columns `id_col`, `time_col`, and `value_col`.

        Returns:
            A Polars DataFrame (or Series) of root mean squared error (RMSE) values, averaged across \
            the rolling-origin folds for each model. Columns represent different models.

        """
        # 1) Compute minimum series length across groups (if multiple series).
        size_df = df.group_by(self.id_col).agg(pl.count(self.time_col).alias("series_length"))
        min_len = size_df["series_length"].min()

        # 2) Derive training/test sizes based on `backtesting_start`
        history_len = int(min_len * self.backtesting_start)  # length of initial training portion
        test_len = min_len - history_len  # length of the test portion

        # 3) Split test portion into n_folds => block_len
        block_len = max(test_len // self.n_folds, 1)

        # 4) Our horizon (h) and step_size both match the block length
        h = block_len
        step_size = block_len

        # 5) Call cross_validation with these derived parameters
        cv_df = self.sf.cross_validation(df=df, h=h, step_size=step_size, n_windows=self.n_folds)

        # Compute SSE (or RMSE, etc.)
        model_names = [m.alias for m in self.sf.models]
        errors = losses.rmse(cv_df, models=model_names, target_col=self.value_col)
        return errors

    def kaboudan_metric(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute the Kaboudan Metric by comparing model errors before and after block-based shuffling.

        This method first calculates a baseline error using `backtest`. Then it applies
        `block_shuffle_by_id` to shuffle each group's rows, re-performs `backtest` on the shuffled data,
        and compares the two sets of errors. The final metric indicates how much performance
        degrades due to the block shuffle.

        Steps:

        1. Compute the baseline RMSE (`sse_before`) for the unshuffled data.
        2. Shuffle the data in blocks (`block_shuffle_by_id`).
        3. Compute the RMSE (`sse_after`) of the shuffled data.
        4. Compute the ratio `sse_before / sse_after` and transform it by `(1 - sqrt(ratio))`.

        If `modified` is True, the resulting metric is clipped at 0 to avoid negative values.

        Args:
            df: A Polars DataFrame with columns for `id_col`, `time_col`, and `value_col`.

        Returns:
            A Polars DataFrame containing columns of Kaboudan Metric values for each model. \
            If `modified` is True, negative values are clipped to zero.

        """
        sse_before = self.backtest(df)

        df_shuffled = self.block_shuffle_by_id(df)
        sse_after = self.backtest(df_shuffled)

        # Compute the metric
        scores = (sse_before.drop(self.id_col) / sse_after.drop(self.id_col)).select(
            sse_before[self.id_col],
            (1 - pl.all().sqrt()).name.keep(),
        )

        if self.agg:
            scores = scores.drop(self.id_col).mean()
        if self.modified:
            # clip to zero
            return scores.with_columns(pl.exclude(self.id_col).clip(lower_bound=0))
        else:
            return scores
