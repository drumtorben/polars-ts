"""Input validation tests — ensure proper Python exceptions, not panics."""

import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)

UNIVARIATE_FUNCTIONS = [
    compute_pairwise_dtw,
    compute_pairwise_ddtw,
    compute_pairwise_wdtw,
    compute_pairwise_msm,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_twe,
]


class TestMissingColumns:
    @pytest.mark.parametrize("fn", UNIVARIATE_FUNCTIONS)
    def test_missing_unique_id_column(self, fn):
        """Missing 'unique_id' column should raise KeyError."""
        df = pl.DataFrame({"id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(KeyError):
            fn(df, df)

    @pytest.mark.parametrize("fn", UNIVARIATE_FUNCTIONS)
    def test_missing_y_column(self, fn):
        """Missing 'y' column should raise KeyError."""
        df = pl.DataFrame({"unique_id": ["A"] * 4, "value": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(KeyError):
            fn(df, df)


class TestEmptyDataFrame:
    @pytest.mark.parametrize("fn", UNIVARIATE_FUNCTIONS)
    def test_empty_dataframe(self, fn):
        """Empty DataFrame should produce empty result, not panic."""
        df = pl.DataFrame({"unique_id": pl.Series([], dtype=pl.String), "y": pl.Series([], dtype=pl.Float64)})
        result = fn(df, df)
        assert result.shape[0] == 0
