from datetime import datetime, timedelta

import polars as pl
import pytest

from polars_ts.decomposition.seasonal_decompose_features import (
    seasonal_decompose_features,
)  # Replace with actual import


@pytest.fixture
def sample_dataframe():
    """Generate a larger sample dataframe for testing with 5 unique IDs and a 2-year datetime range."""
    # Create a datetime range for 2 years (730 days)
    date_range = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(730)]

    # Create data with 5 unique ids, cycling through them
    unique_ids = [i % 5 + 1 for i in range(730)]  # Cycle through 5 unique IDs

    # Generate some simple time series data with slight variations
    y_data = [round(5 + (i % 10) + (i // 50), 2) for i in range(730)]  # Some changing time series data

    data = {
        "id_col": unique_ids,
        "time_col": date_range,
        "target_col": y_data,
    }

    return pl.DataFrame(data)


# Test for 'simple' mode
def test_seasonal_decompose_simple_mode(sample_dataframe):
    # Call the function with simple mode
    result = seasonal_decompose_features(
        df=sample_dataframe, id_col="id_col", time_col="time_col", target_col="target_col", ts_freq=24, mode="simple"
    )

    # Ensure the output DataFrame has the expected columns
    assert "trend_strength" in result.columns
    assert "seasonal_strength" in result.columns
    assert "resid_var" in result.columns


# Test for 'mstl' mode with missing seasonal_freqs
def test_seasonal_decompose_mstl_missing_seasonal_freqs(sample_dataframe):
    # Call the function with 'mstl' mode but no seasonal_freqs
    with pytest.raises(ValueError):
        seasonal_decompose_features(
            df=sample_dataframe,
            id_col="id_col",
            time_col="time_col",
            target_col="target_col",
            ts_freq=24,
            mode="mstl",  # seasonal_freqs is missing
        )


# Test for missing columns in input DataFrame
def test_seasonal_decompose_missing_column(sample_dataframe):
    # Remove 'target_col' to simulate missing column
    df_missing_col = sample_dataframe.drop("target_col")

    with pytest.raises(AssertionError):
        seasonal_decompose_features(
            df=df_missing_col, id_col="id_col", time_col="time_col", target_col="target_col", ts_freq=24, mode="simple"
        )


# Test for invalid mode value
def test_seasonal_decompose_invalid_mode(sample_dataframe):
    # Call with an invalid mode
    with pytest.raises(AssertionError, match='Must Pick a mode "mstl" or "simple" to specify type of decomposition...'):
        seasonal_decompose_features(
            df=sample_dataframe,
            id_col="id_col",
            time_col="time_col",
            target_col="target_col",
            ts_freq=24,
            mode="invalid_mode",
        )


# Test for seasonal_freqs of the wrong type (not a list)
def test_seasonal_decompose_invalid_seasonal_freq_type(sample_dataframe):
    # Call with a non-list seasonal_freqs
    with pytest.raises(AssertionError):
        seasonal_decompose_features(
            df=sample_dataframe,
            id_col="id_col",
            time_col="time_col",
            target_col="target_col",
            ts_freq=24,
            seasonal_freqs="daily",  # Invalid type
            mode="mstl",
        )


# # Test for seasonal_freqs with empty list
def test_seasonal_decompose_empty_seasonal_freqs(sample_dataframe):
    # Call with an empty list of seasonal frequencies
    with pytest.raises(AssertionError):
        seasonal_decompose_features(
            df=sample_dataframe,
            id_col="id_col",
            time_col="time_col",
            target_col="target_col",
            ts_freq=7,
            seasonal_freqs=[],  # Empty list
            mode="mstl",
        )


# # Unit test for valid frequencies
# @pytest.mark.parametrize("valid_freq", [12, 52,4,24])
# def test_valid_frequencies(sample_dataframe,valid_freq):

#     try:
#         seasonal_decompose_features(
#             df=sample_dataframe,
#             id_col="id_col",
#             time_col="time_col",
#             target_col="target_col",
#             ts_freq = valid_freq,  # Pass valid frequency
#             seasonal_freqs = [valid_freq],
#             mode="mstl"
#         )
#     except AssertionError as e:
#         pytest.fail(f"Unexpected AssertionError: {str(e)}")

# # Unit test for invalid frequency (not in freq_mapper)
# def test_invalid_frequency(sample_dataframe):

#     with pytest.raises(AssertionError):
#         seasonal_decompose_features(
#             df=sample_dataframe,
#            id_col="id_col",
#             time_col="time_col",
#             target_col="target_col",
#             ts_freq=5,  # Invalid frequency not in the freq_mapper
#             seasonal_freqs = [5],
#             mode="mstl"
#         )
