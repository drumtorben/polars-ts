import pytest
import polars as pl
from datetime import datetime, timedelta
from polars_ts.decomposition.fourier_decomposition import fourier_decomposition  # Update this with the correct import


@pytest.fixture
def sample_df():
    """Generate a larger sample dataframe for testing with 5 unique IDs and a 2-year datetime range."""
    # Create a datetime range for 2 years (730 days)
    date_range = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(730)]

    # Create data with 5 unique ids, cycling through them
    unique_ids = [i % 5 + 1 for i in range(730)]  # Cycle through 5 unique IDs

    # Generate some simple time series data with slight variations
    y_data = [round(5 + (i % 10) + (i // 50), 2) for i in range(730)]  # Some changing time series data

    data = {
        "unique_id": unique_ids,
        "ds": date_range,
        "y": y_data,
    }

    return pl.DataFrame(data)


def test_fourier_decomposition_basic(sample_df):
    """Test basic functionality of fourier_decomposition."""
    result = fourier_decomposition(sample_df, ts_freq=7)

    # Check if the resulting DataFrame contains the expected columns
    expected_columns = ["unique_id", "ds", "y", "trend", "seasonal", "resid"]
    assert set(result.columns) == set(expected_columns), (
        f"Expected columns: {expected_columns}, but got: {result.columns}"
    )

    # Check if the result has the same number of rows
    assert result.height == 700, "Resulting DataFrame has incorrect number of rows"


def test_fourier_decomposition_components(sample_df):
    # Run Fourier decomposition on the sample DataFrame
    result = fourier_decomposition(sample_df, ts_freq=7)

    # Extract the relevant columns for comparison
    trend_col = result["trend"]
    seasonal_col = result["seasonal"]
    resid_col = result["resid"]

    # Assert that the trend, seasonal, and residual columns are not equal to each other
    assert not (trend_col == seasonal_col).all(), "Trend and Seasonal components should not be equal"
    assert not (trend_col == resid_col).all(), "Trend and Residual components should not be equal"
    assert not (seasonal_col == resid_col).all(), "Seasonal and Residual components should not be equal"

    # Optionally, check that all the components are not equal to the original target column (y)
    target_col = result["y"]
    assert not (trend_col == target_col).all(), "Trend should not be equal to the original target"
    assert not (seasonal_col == target_col).all(), "Seasonal should not be equal to the original target"
    assert not (resid_col == target_col).all(), "Residual should not be equal to the original target"
