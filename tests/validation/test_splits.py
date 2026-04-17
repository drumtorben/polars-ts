"""Tests for time series cross-validation splitters."""

from datetime import date

import polars as pl
import pytest

from polars_ts.validation.splits import (
    expanding_window_cv,
    rolling_origin_cv,
    sliding_window_cv,
)


def _make_df() -> pl.DataFrame:
    """Two groups, 10 time steps each."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "ds": [date(2024, 1, i) for i in range(1, 11)] * 2,
            "y": list(range(1, 11)) + list(range(101, 111)),
        }
    )


# --- expanding_window_cv ---


def test_expanding_fold_count():
    df = _make_df()
    folds = list(expanding_window_cv(df, n_splits=3, horizon=2))
    assert len(folds) == 3


def test_expanding_train_grows():
    df = _make_df()
    folds = list(expanding_window_cv(df, n_splits=3, horizon=2, step=1))
    train_sizes = [len(train) for train, _ in folds]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1]


def test_expanding_test_size_constant():
    df = _make_df()
    horizon = 2
    n_groups = 2
    folds = list(expanding_window_cv(df, n_splits=3, horizon=horizon))
    for _, test in folds:
        assert len(test) == horizon * n_groups


def test_expanding_no_overlap():
    df = _make_df()
    for train, test in expanding_window_cv(df, n_splits=3, horizon=2):
        train_times = set(train["ds"].unique().to_list())
        test_times = set(test["ds"].unique().to_list())
        assert train_times.isdisjoint(test_times)


def test_expanding_temporal_order():
    df = _make_df()
    for train, test in expanding_window_cv(df, n_splits=3, horizon=2):
        assert train["ds"].max() < test["ds"].min()


# --- sliding_window_cv ---


def test_sliding_train_size_constant():
    df = _make_df()
    train_size = 4
    n_groups = 2
    folds = list(sliding_window_cv(df, n_splits=3, train_size=train_size, horizon=2))
    for train, _ in folds:
        assert len(train) == train_size * n_groups


def test_sliding_train_slides():
    df = _make_df()
    folds = list(sliding_window_cv(df, n_splits=3, train_size=4, horizon=2, step=1))
    min_times = [train["ds"].min() for train, _ in folds]
    for i in range(1, len(min_times)):
        assert min_times[i] > min_times[i - 1]


def test_sliding_no_overlap():
    df = _make_df()
    for train, test in sliding_window_cv(df, n_splits=3, train_size=4, horizon=2):
        train_times = set(train["ds"].unique().to_list())
        test_times = set(test["ds"].unique().to_list())
        assert train_times.isdisjoint(test_times)


# --- rolling_origin_cv ---


def test_rolling_origin_expanding_mode():
    df = _make_df()
    folds_exp = list(expanding_window_cv(df, n_splits=3, horizon=2))
    folds_ro = list(rolling_origin_cv(df, n_splits=3, horizon=2, fixed_train_size=None))
    for (t1, s1), (t2, s2) in zip(folds_exp, folds_ro, strict=False):
        assert t1["y"].to_list() == t2["y"].to_list()
        assert s1["y"].to_list() == s2["y"].to_list()


def test_rolling_origin_sliding_mode():
    df = _make_df()
    folds_sl = list(sliding_window_cv(df, n_splits=3, train_size=4, horizon=2))
    folds_ro = list(rolling_origin_cv(df, n_splits=3, horizon=2, fixed_train_size=4))
    for (t1, s1), (t2, s2) in zip(folds_sl, folds_ro, strict=False):
        assert t1["y"].to_list() == t2["y"].to_list()
        assert s1["y"].to_list() == s2["y"].to_list()


# --- gap and step ---


def test_gap_parameter():
    df = _make_df()
    gap = 2
    for train, test in expanding_window_cv(df, n_splits=2, horizon=1, gap=gap):
        train_max = train["ds"].max()
        test_min = test["ds"].min()
        # There should be `gap` time steps between train end and test start
        all_times = sorted(df["ds"].unique().to_list())
        train_max_idx = all_times.index(train_max)
        test_min_idx = all_times.index(test_min)
        assert test_min_idx - train_max_idx == gap + 1


def test_step_parameter():
    df = _make_df()
    step = 2
    folds = list(expanding_window_cv(df, n_splits=3, horizon=1, step=step))
    # Train sizes should increase by step * n_groups each fold
    train_sizes = [len(train) for train, _ in folds]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] - train_sizes[i - 1] == step * 2  # 2 groups


# --- groups ---


def test_groups_in_every_fold():
    df = _make_df()
    for train, test in expanding_window_cv(df, n_splits=3, horizon=2):
        assert set(train["unique_id"].unique().to_list()) == {"A", "B"}
        assert set(test["unique_id"].unique().to_list()) == {"A", "B"}


def test_single_group():
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": [date(2024, 1, i) for i in range(1, 11)],
            "y": list(range(10)),
        }
    )
    folds = list(expanding_window_cv(df, n_splits=3, horizon=2))
    assert len(folds) == 3
    for _train, test in folds:
        assert len(test) == 2


# --- validation ---


def test_insufficient_data_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="Not enough"):
        list(expanding_window_cv(df, n_splits=20, horizon=5))


def test_n_splits_one():
    df = _make_df()
    folds = list(expanding_window_cv(df, n_splits=1, horizon=2))
    assert len(folds) == 1


def test_unsorted_input():
    df = _make_df().sample(fraction=1.0, shuffle=True, seed=42)
    folds = list(expanding_window_cv(df, n_splits=3, horizon=2))
    assert len(folds) == 3
    for train, test in folds:
        assert train["ds"].max() < test["ds"].min()


# --- namespace ---


def test_expanding_via_namespace():
    from polars_ts.metrics import Metrics  # noqa: F401

    df = _make_df()
    folds = list(df.pts.expanding_window_cv(n_splits=2, horizon=2))
    assert len(folds) == 2


def test_sliding_via_namespace():
    from polars_ts.metrics import Metrics  # noqa: F401

    df = _make_df()
    folds = list(df.pts.sliding_window_cv(n_splits=2, train_size=4, horizon=2))
    assert len(folds) == 2
