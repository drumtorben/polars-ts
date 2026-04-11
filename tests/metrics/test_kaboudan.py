from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ts.metrics.kaboudan import Kaboudan


# Sample data for testing
@pytest.fixture
def sample_df():
    data = {
        "unique_id": ["A"] * 10 + ["B"] * 10,
        "ds": list(range(10)) * 2,
        "y": [i + (j * 10) for j in range(2) for i in range(10)],
    }
    return pl.DataFrame(data)


# Mock StatsForecast
@pytest.fixture
def mock_statsforecast():
    mock_sf = MagicMock()
    # Include 'y' in the cross_validation return to match the backtest requirements
    mock_sf.cross_validation.return_value = pl.DataFrame(
        {
            "unique_id": ["A"] * 5 + ["B"] * 5,
            "y": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "model_1": [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8],
            "model_2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    mock_sf.models = [MagicMock(alias="model_1"), MagicMock(alias="model_2")]
    return mock_sf


def test_initialization(mock_statsforecast):
    kaboudan = Kaboudan(
        sf=mock_statsforecast,
        backtesting_start=0.8,
        n_folds=4,
        block_size=5,
        seed=123,
        id_col="unique_id",
        time_col="ds",
        value_col="y",
        modified=False,
        agg=True,
    )
    assert kaboudan.sf == mock_statsforecast
    assert kaboudan.backtesting_start == 0.8
    assert kaboudan.n_folds == 4
    assert kaboudan.block_size == 5
    assert kaboudan.seed == 123
    assert kaboudan.id_col == "unique_id"
    assert kaboudan.time_col == "ds"
    assert kaboudan.value_col == "y"
    assert not kaboudan.modified
    assert kaboudan.agg


def test_block_shuffle_by_id(sample_df):
    kaboudan = Kaboudan(
        sf=None,
        backtesting_start=0.7,
        n_folds=5,
        block_size=2,
        seed=42,
        id_col="unique_id",
        time_col="ds",
        value_col="y",
    )
    shuffled_df = kaboudan.block_shuffle_by_id(sample_df)

    # Check that the unique_id and ds columns are still present
    assert "unique_id" in shuffled_df.columns
    assert "ds" in shuffled_df.columns
    assert "y" in shuffled_df.columns

    # Check that all unique_ids are still present
    original_ids = set(sample_df["unique_id"].unique())
    shuffled_ids = set(shuffled_df["unique_id"].unique())
    assert original_ids == shuffled_ids

    # Ensure that within each unique_id, the number of rows remains the same
    for uid in original_ids:
        original_count = sample_df.filter(pl.col("unique_id") == uid).height
        shuffled_count = shuffled_df.filter(pl.col("unique_id") == uid).height
        assert original_count == shuffled_count


def test_split_in_blocks_by_id(sample_df):
    kaboudan = Kaboudan(
        sf=None,
        backtesting_start=0.7,
        n_folds=5,
        block_size=2,
        seed=42,
        id_col="unique_id",
        time_col="ds",
        value_col="y",
    )
    split_df = kaboudan.split_in_blocks_by_id(sample_df)

    # Check that 'block' column is added
    assert "block" in split_df.columns

    # Check that block numbers are between 1 and n_folds
    assert split_df["block"].min() >= 1
    assert split_df["block"].max() <= kaboudan.n_folds

    # Verify block assignments for a sample group
    group_a = split_df.filter(pl.col("unique_id") == "A").sort("ds")
    expected_blocks = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert group_a["block"].to_list() == expected_blocks


def test_backtest(sample_df, mock_statsforecast):
    kaboudan = Kaboudan(
        sf=mock_statsforecast,
        backtesting_start=0.7,
        n_folds=2,
        block_size=3,
        seed=42,
    )
    errors = kaboudan.backtest(sample_df)

    # Check that cross_validation was called with correct parameters
    test_len = 3
    block_len = max(test_len // kaboudan.n_folds, 1)  # 1
    h = block_len
    step_size = block_len
    mock_statsforecast.cross_validation.assert_called_once_with(
        df=sample_df, h=h, step_size=step_size, n_windows=kaboudan.n_folds
    )

    # Check that errors are computed correctly
    expected_errors = pl.DataFrame(
        {
            "unique_id": ["A", "B"],
            "model_1": [11.371455491712572, 21.437817053049034],
            "model_2": [11.470832576583096, 21.53764146790451],
        }
    )
    # Verify the metrics
    assert_frame_equal(errors, expected_errors)


def test_kaboudan_metric(sample_df, mock_statsforecast):
    kaboudan = Kaboudan(
        sf=mock_statsforecast, backtesting_start=0.7, n_folds=2, block_size=3, seed=42, modified=True, agg=False
    )

    with (
        patch.object(kaboudan, "block_shuffle_by_id") as mock_shuffle,
        patch.object(kaboudan, "backtest") as mock_backtest,
    ):
        # Setup mock_backtest responses
        # First call: sse_before
        # Second call: sse_after
        mock_backtest.side_effect = [
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [1.0, 2.0],
                    "model_2": [1.5, 2.5],
                }
            ),
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [2.0, 4.0],
                    "model_2": [3.0, 5.0],
                }
            ),
        ]

        # Setup mock_shuffle to return shuffled DataFrame
        mock_shuffle.return_value = sample_df

        metrics = kaboudan.kaboudan_metric(sample_df)

        # Expected ratio = before / after
        # model_1: [1.0 / 2.0, 2.0 / 4.0] = [0.5, 0.5]
        # model_2: [1.5 / 3.0, 2.5 / 5.0] = [0.5, 0.5]
        # 1 - sqrt(0.5) ≈ 0.2928932188134524 for each
        expected_metrics = pl.DataFrame(
            {
                "unique_id": ["A", "B"],
                "model_1": [0.2928932188134524, 0.2928932188134524],
                "model_2": [0.2928932188134524, 0.2928932188134524],
            }
        )

        # Verify the metrics
        assert_frame_equal(metrics, expected_metrics)

        # Ensure block_shuffle_by_id and backtest were called twice
        assert mock_shuffle.call_count == 1
        assert mock_backtest.call_count == 2


def test_kaboudan_metric_modified_false(sample_df, mock_statsforecast):
    kaboudan = Kaboudan(
        sf=mock_statsforecast, backtesting_start=0.7, n_folds=2, block_size=3, seed=42, modified=False, agg=False
    )

    with (
        patch.object(kaboudan, "block_shuffle_by_id") as mock_shuffle,
        patch.object(kaboudan, "backtest") as mock_backtest,
    ):
        # Setup mock_backtest responses
        mock_backtest.side_effect = [
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [1.0, 2.0],
                    "model_2": [1.5, 2.5],
                }
            ),
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [2.0, 4.0],
                    "model_2": [3.0, 5.0],
                }
            ),
        ]

        # Setup mock_shuffle to return shuffled DataFrame
        mock_shuffle.return_value = sample_df

        metrics = kaboudan.kaboudan_metric(sample_df)

        # Expected ratio = before / after
        # model_1: [1.0 / 2.0, 2.0 / 4.0] = [0.5, 0.5]
        # model_2: [1.5 / 3.0, 2.5 / 5.0] = [0.5, 0.5]
        # 1 - sqrt(0.5) ≈ 0.2928932188134524 for each
        expected_metrics = pl.DataFrame(
            {
                "unique_id": ["A", "B"],
                "model_1": [0.2928932188134524, 0.2928932188134524],
                "model_2": [0.2928932188134524, 0.2928932188134524],
            }
        )

        # Verify the metrics
        assert_frame_equal(metrics, expected_metrics)


def test_kaboudan_metric_aggregation(sample_df, mock_statsforecast):
    kaboudan = Kaboudan(
        sf=mock_statsforecast, backtesting_start=0.7, n_folds=2, block_size=3, seed=42, modified=True, agg=True
    )

    with (
        patch.object(kaboudan, "block_shuffle_by_id") as mock_shuffle,
        patch.object(kaboudan, "backtest") as mock_backtest,
    ):
        # Setup mock_backtest responses
        mock_backtest.side_effect = [
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [1.0, 2.0],
                    "model_2": [1.5, 2.5],
                }
            ),
            pl.DataFrame(
                {
                    "unique_id": ["A", "B"],
                    "model_1": [2.0, 4.0],
                    "model_2": [3.0, 5.0],
                }
            ),
        ]

        # Setup mock_shuffle to return shuffled DataFrame
        mock_shuffle.return_value = sample_df

        metrics = kaboudan.kaboudan_metric(sample_df)

        # Expected ratio = before / after
        # model_1: [1.0 / 2.0, 2.0 / 4.0] = [0.5, 0.5]
        # model_2: [1.5 / 3.0, 2.5 / 5.0] = [0.5, 0.5]
        # 1 - sqrt(0.5) ≈ 0.292893 for each
        # Aggregated means: 0.292893 for each model
        expected_metrics = pl.DataFrame(
            {
                "model_1": [0.2928932188134524],
                "model_2": [0.2928932188134524],
            }
        )

        # Verify the metrics
        assert_frame_equal(metrics, expected_metrics)

        # Ensure block_shuffle_by_id and backtest were called twice
        assert mock_shuffle.call_count == 1
        assert mock_backtest.call_count == 2


# Additional tests can be added to cover edge cases, such as empty DataFrame, single group, etc.


def test_empty_dataframe():
    kaboudan = Kaboudan(
        sf=None,
        backtesting_start=0.7,
        n_folds=5,
        block_size=2,
    )
    empty_df = pl.DataFrame(
        {
            "unique_id": [],
            "ds": [],
            "y": [],
        }
    )
    with pytest.raises(TypeError):
        kaboudan.backtest(empty_df)


def test_single_group(sample_df):
    single_group_df = sample_df.filter(pl.col("unique_id") == "A")
    kaboudan = Kaboudan(
        sf=None,
        backtesting_start=0.7,
        n_folds=2,
        block_size=3,
    )
    split_df = kaboudan.split_in_blocks_by_id(single_group_df)
    assert "block" in split_df.columns
    assert split_df["block"].min() >= 1
    assert split_df["block"].max() <= kaboudan.n_folds
