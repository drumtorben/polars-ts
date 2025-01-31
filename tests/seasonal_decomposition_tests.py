import pytest
import polars as pl
from polars_ts.decomposition.seasonal_decomposition import seasonal_decomposition  # Make sure to import your actual function

# Helper function to create a sample DataFrame
def create_sample_df():
    return pl.DataFrame({
        "unique_id": ["A", "A", "A", "B", "B", "B"],
        "ds": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-02-01", "2020-03-01"],
        "y": [10, 15, 20, 5, 7, 9]
    }).with_columns(pl.col("ds").str.to_date("%Y-%m-%d"))

# Test: Valid case with additive method
def test_valid_additive():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3, method="additive")
  #  assert result.shape[0] == df.shape[0], "Resulting DataFrame should have the same number of rows"
    assert "trend" in result.columns, "Result should contain 'trend' column"
    assert "seasonal_3" in result.columns, "Result should contain 'seasonal_3' column"
    assert "resid" in result.columns, "Result should contain 'resid' column"

# Test: Invalid method argument
def test_invalid_method():
    df = create_sample_df()
    with pytest.raises(ValueError, match="Invalid method 'invalid'. Expected 'additive' or 'multiplicative'."):
        seasonal_decomposition(df, freq=3, method="invalid")

# Test: Missing column
def test_missing_column():
    df = create_sample_df().drop("y")  # Drop the target_col
    with pytest.raises(KeyError, match="Column 'y' is missing from the DataFrame."):
        seasonal_decomposition(df, freq=3)
    
# Test: Invalid frequency
def test_invalid_frequency():
    df = create_sample_df()
    with pytest.raises(ValueError, match="Invalid frequency '0'. Frequency must be greater than 1."):
        seasonal_decomposition(df, freq=0)

# Test: Ensure exception is raised for missing columns
def test_missing_time_column():
    df = create_sample_df().drop("ds")  # Drop the time column
    with pytest.raises(KeyError, match="Column 'ds' is missing from the DataFrame."):
        seasonal_decomposition(df, freq=3)

# Test: Ensure method works for multiplicative
def test_valid_multiplicative():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3, method="multiplicative")
   # assert result.shape[0] == df.shape[0], "Resulting DataFrame should have the same number of rows"
    assert "trend" in result.columns, "Result should contain 'trend' column"
    assert "seasonal_3" in result.columns, "Result should contain 'seasonal_3' column"
    assert "resid" in result.columns, "Result should contain 'resid' column"

