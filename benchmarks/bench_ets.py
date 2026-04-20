"""Benchmarks for exponential smoothing (Rust vs Python)."""

import numpy as np
import pytest

from polars_ts.models.exponential_smoothing import (
    _hw_python,
    _ses_python,
)


def _make_values(n: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, n).cumsum() + 100).tolist()


def _make_seasonal(n: int, m: int = 12, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    trend = np.linspace(10, 30, n)
    seasonal = 5.0 * np.sin(2 * np.pi * np.arange(n) / m)
    noise = rng.normal(0, 0.5, n)
    return (trend + seasonal + noise).tolist()


# --- Fixtures ---


@pytest.fixture(scope="module")
def vals_1k():
    return _make_values(1000)


@pytest.fixture(scope="module")
def vals_10k():
    return _make_values(10000)


@pytest.fixture(scope="module")
def seasonal_1k():
    return _make_seasonal(1000, m=12)


@pytest.fixture(scope="module")
def seasonal_10k():
    return _make_seasonal(10000, m=12)


# --- SES benchmarks ---


def test_ses_rust_1k(benchmark, vals_1k):
    from polars_ts_rs import ets_ses

    benchmark(ets_ses, vals_1k, 0.3, 12)


def test_ses_python_1k(benchmark, vals_1k):
    benchmark(_ses_python, vals_1k, 0.3, 12)


def test_ses_rust_10k(benchmark, vals_10k):
    from polars_ts_rs import ets_ses

    benchmark(ets_ses, vals_10k, 0.3, 12)


def test_ses_python_10k(benchmark, vals_10k):
    benchmark(_ses_python, vals_10k, 0.3, 12)


# --- Holt-Winters benchmarks ---


def test_hw_rust_1k(benchmark, seasonal_1k):
    from polars_ts_rs import ets_holt_winters

    benchmark(ets_holt_winters, seasonal_1k, 0.3, 0.1, 0.1, 12, True, 24)


def test_hw_python_1k(benchmark, seasonal_1k):
    benchmark(_hw_python, seasonal_1k, 0.3, 0.1, 0.1, 12, True, 24)


def test_hw_rust_10k(benchmark, seasonal_10k):
    from polars_ts_rs import ets_holt_winters

    benchmark(ets_holt_winters, seasonal_10k, 0.3, 0.1, 0.1, 12, True, 24)


def test_hw_python_10k(benchmark, seasonal_10k):
    benchmark(_hw_python, seasonal_10k, 0.3, 0.1, 0.1, 12, True, 24)
