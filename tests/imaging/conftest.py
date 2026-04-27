import polars as pl
import pytest


@pytest.fixture
def sample_data():
    """Two series: A ascending (10 points), B alternating (10 points)."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "y": [float(i) for i in range(10)] + [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
