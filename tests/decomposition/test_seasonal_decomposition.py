import polars as pl
import pytest

from polars_ts.decomposition.seasonal_decomposition import (
    seasonal_decomposition,
)  # Make sure to import your actual function


# Helper function to create a sample DataFrame
def create_sample_df():
    return pl.DataFrame(
        {
            "unique_id": ["A", "A", "A", "B", "B", "B"],
            "ds": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-02-01", "2020-03-01"],
            "y": [10, 15, 20, 5, 7, 9],
        }
    ).with_columns(pl.col("ds").str.to_date("%Y-%m-%d"))


# Test: Valid case with additive method
def test_valid_additive():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3, method="additive")
    assert "trend" in result.columns, "Result should contain 'trend' column"
    assert "seasonal" in result.columns, "Result should contain 'seasonal' column"
    assert "resid" in result.columns, "Result should contain 'resid' column"


# Test: Invalid method argument
def test_invalid_method():
    df = create_sample_df()
    with pytest.raises(ValueError, match="Invalid method 'invalid'. Expected 'additive' or 'multiplicative'."):
        seasonal_decomposition(df, freq=3, method="invalid")


# Test: Missing column
def test_missing_column():
    df = create_sample_df().drop("y")  # Drop the target_col
    with pytest.raises(KeyError, match="Columns {'y'} are missing from the DataFrame."):
        seasonal_decomposition(df, freq=3)


# Test: Invalid frequency
def test_invalid_frequency():
    df = create_sample_df()
    with pytest.raises(ValueError, match="Invalid frequency '0'. Frequency must be greater than 1."):
        seasonal_decomposition(df, freq=0)


# Test: Ensure exception is raised for missing columns
def test_missing_time_column():
    df = create_sample_df().drop("ds")  # Drop the time column
    with pytest.raises(KeyError, match="Columns {'ds'} are missing from the DataFrame."):
        seasonal_decomposition(df, freq=3)


# Test: Ensure method works for multiplicative
def test_valid_multiplicative():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3, method="multiplicative")
    assert "trend" in result.columns, "Result should contain 'trend' column"
    assert "seasonal" in result.columns, "Result should contain 'seasonal' column"
    assert "resid" in result.columns, "Result should contain 'resid' column"


# Test: anomaly_threshold adds is_anomaly column
def test_anomaly_threshold_adds_column():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3, anomaly_threshold=2.0)
    assert "is_anomaly" in result.columns, "Result should contain 'is_anomaly' column"
    assert result["is_anomaly"].dtype == pl.Boolean


# Test: no anomaly column when threshold is not set
def test_no_anomaly_column_without_threshold():
    df = create_sample_df()
    result = seasonal_decomposition(df, freq=3)
    assert "is_anomaly" not in result.columns


# Test: anomaly_threshold flags outliers correctly
def test_anomaly_threshold_flags_outliers():
    # Create a series with a clear outlier
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 12,
            "ds": [f"2020-{m:02d}-01" for m in range(1, 13)],
            "y": [10.0, 12.0, 11.0, 10.0, 11.0, 100.0, 10.0, 12.0, 11.0, 10.0, 11.0, 10.0],
        }
    ).with_columns(pl.col("ds").str.to_date("%Y-%m-%d"))
    result = seasonal_decomposition(df, freq=3, anomaly_threshold=1.5)
    assert "is_anomaly" in result.columns
    # At least one anomaly should be flagged
    assert result["is_anomaly"].sum() >= 1
