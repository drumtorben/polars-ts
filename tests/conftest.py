import polars as pl
import pytest


@pytest.fixture
def two_series():
    """Two simple time series A and B that differ by one point."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0],
        }
    )


@pytest.fixture
def three_series():
    """Three time series: A ascending, B similar to A, C reversed."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


@pytest.fixture
def identical_series():
    """Two identical time series."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def shifted_series():
    """Two series where one is a shifted version — constraints matter here."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8,
            "y": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def single_series():
    """Return a single time series — no pairs to compare."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def int_id_series():
    """Time series with integer unique_id."""
    return pl.DataFrame(
        {
            "unique_id": [1] * 4 + [2] * 4,
            "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
