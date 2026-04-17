"""Time series cross-validation splitters (group-aware, temporal)."""

from __future__ import annotations

from collections.abc import Generator

import polars as pl


def expanding_window_cv(
    df: pl.DataFrame,
    n_splits: int = 5,
    horizon: int = 1,
    step: int = 1,
    gap: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """Expand-window time series cross-validation.

    The training window starts at the beginning and grows by ``step``
    time steps each fold. The test window is always ``horizon`` steps.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    n_splits
        Number of (train, test) folds to generate.
    horizon
        Number of time steps in each test fold.
    step
        Number of time steps the split point advances between folds.
    gap
        Number of time steps between training end and test start.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Yields
    ------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, test_df) for each fold.

    """
    yield from _rolling_origin_splits(
        df,
        n_splits=n_splits,
        horizon=horizon,
        step=step,
        gap=gap,
        fixed_train_size=None,
        id_col=id_col,
        time_col=time_col,
    )


def sliding_window_cv(
    df: pl.DataFrame,
    n_splits: int = 5,
    train_size: int = 10,
    horizon: int = 1,
    step: int = 1,
    gap: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """Slide fixed-size window time series cross-validation.

    Parameters
    ----------
    df
        Input DataFrame.
    n_splits
        Number of folds.
    train_size
        Fixed number of time steps in each training window.
    horizon
        Number of time steps in each test fold.
    step
        How many time steps to advance between folds.
    gap
        Gap between train end and test start.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Yields
    ------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, test_df) for each fold.

    """
    yield from _rolling_origin_splits(
        df,
        n_splits=n_splits,
        horizon=horizon,
        step=step,
        gap=gap,
        fixed_train_size=train_size,
        id_col=id_col,
        time_col=time_col,
    )


def rolling_origin_cv(
    df: pl.DataFrame,
    n_splits: int = 5,
    initial_train_size: int | None = None,
    horizon: int = 1,
    step: int = 1,
    gap: int = 0,
    fixed_train_size: int | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """Perform general rolling-origin cross-validation.

    Configurable as expanding (``fixed_train_size=None``) or sliding
    (``fixed_train_size=k``) window CV.

    Parameters
    ----------
    df
        Input DataFrame.
    n_splits
        Number of folds.
    initial_train_size
        Minimum time steps for the first training fold. If ``None``,
        computed automatically from other parameters.
    horizon
        Test window size in time steps.
    step
        Advance between successive split points.
    gap
        Gap between train and test.
    fixed_train_size
        If set, training window is always this size (sliding).
        If ``None``, training window expands.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Yields
    ------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, test_df) for each fold.

    """
    yield from _rolling_origin_splits(
        df,
        n_splits=n_splits,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=step,
        gap=gap,
        fixed_train_size=fixed_train_size,
        id_col=id_col,
        time_col=time_col,
    )


def _rolling_origin_splits(
    df: pl.DataFrame,
    n_splits: int,
    horizon: int,
    step: int,
    gap: int,
    fixed_train_size: int | None = None,
    initial_train_size: int | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """Produce n_splits (train, test) folds from a time series DataFrame."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if fixed_train_size is not None and fixed_train_size < 1:
        raise ValueError("fixed_train_size must be >= 1")

    sorted_df = df.sort(id_col, time_col)
    time_points = sorted_df[time_col].unique().sort().to_list()
    n_times = len(time_points)

    # Compute initial_train_size if not given
    if initial_train_size is None:
        initial_train_size = n_times - (n_splits - 1) * step - gap - horizon
    if initial_train_size < 1:
        raise ValueError(
            f"Not enough time points ({n_times}) for {n_splits} folds "
            f"with horizon={horizon}, step={step}, gap={gap}"
        )

    for i in range(n_splits):
        split_end = initial_train_size + i * step

        # Training window
        if fixed_train_size is not None:
            train_start = split_end - fixed_train_size
            if train_start < 0:
                raise ValueError(f"fixed_train_size={fixed_train_size} too large for fold {i}")
        else:
            train_start = 0

        train_times = time_points[train_start:split_end]

        # Test window (after gap)
        test_start = split_end + gap
        test_end = test_start + horizon
        if test_end > n_times:
            raise ValueError(f"Not enough time points for fold {i}: need index {test_end}, have {n_times}")
        test_times = time_points[test_start:test_end]

        train_df = sorted_df.filter(pl.col(time_col).is_in(train_times))
        test_df = sorted_df.filter(pl.col(time_col).is_in(test_times))

        yield train_df, test_df
