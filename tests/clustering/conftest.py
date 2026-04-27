import polars as pl
import pytest


@pytest.fixture
def well_separated_data():
    """Six series in two well-separated groups (ascending vs descending)."""
    ascending = [1.0, 2.0, 3.0, 4.0]
    descending = [4.0, 3.0, 2.0, 1.0]
    return pl.DataFrame(
        {
            "unique_id": (["A1"] * 4 + ["A2"] * 4 + ["A3"] * 4 + ["B1"] * 4 + ["B2"] * 4 + ["B3"] * 4),
            "y": (
                ascending
                + [1.0, 2.1, 3.0, 4.1]
                + [1.0, 1.9, 3.1, 4.0]
                + descending
                + [4.1, 3.0, 2.0, 0.9]
                + [3.9, 3.1, 1.9, 1.0]
            ),
        }
    )
