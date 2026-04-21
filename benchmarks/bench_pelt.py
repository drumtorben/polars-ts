"""Benchmarks for PELT changepoint detection."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _make_series(n: int = 1000, changepoints: int = 5, seed: int = 42) -> pl.DataFrame:
    """Generate a series with known mean shifts."""
    rng = np.random.default_rng(seed)
    n_segments = changepoints + 1
    values: list[float] = []
    for i in range(n_segments):
        # Last segment absorbs remainder so total length == n
        seg_len = n // n_segments + (1 if i < n % n_segments else 0)
        mean = rng.uniform(-10, 10)
        seg = rng.normal(mean, 1.0, seg_len).tolist()
        values.extend(seg)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values,
        }
    )


@pytest.fixture(scope="module")
def df_1k() -> pl.DataFrame:
    return _make_series(n=1000, changepoints=5)


@pytest.fixture(scope="module")
def df_10k() -> pl.DataFrame:
    return _make_series(n=10000, changepoints=10)


def test_pelt_1k(benchmark, df_1k):
    from polars_ts.changepoint.pelt import pelt

    result = benchmark(pelt, df_1k)
    assert result.height >= 0


def test_pelt_10k(benchmark, df_10k):
    from polars_ts.changepoint.pelt import pelt

    result = benchmark(pelt, df_10k)
    assert result.height >= 0


def test_pelt_python_1k(benchmark, df_1k):
    from polars_ts.changepoint.pelt import _pelt_python

    result = benchmark(_pelt_python, df_1k, "y", "unique_id", "ds", "mean", None, 2)
    assert result.height >= 0


def test_pelt_python_10k(benchmark, df_10k):
    from polars_ts.changepoint.pelt import _pelt_python

    result = benchmark(_pelt_python, df_10k, "y", "unique_id", "ds", "mean", None, 2)
    assert result.height >= 0
