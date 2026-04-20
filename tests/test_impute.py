"""Tests for missing value imputation (#60)."""

from datetime import date

import polars as pl
import pytest

from polars_ts.imputation import impute


def _make_df_with_nulls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": [date(2024, 1, i + 1) for i in range(6)] * 2,
            "y": [1.0, None, 3.0, None, 5.0, 6.0, 10.0, None, None, 40.0, 50.0, 60.0],
        }
    )


class TestImpute:
    def test_forward_fill(self):
        result = impute(_make_df_with_nulls(), method="forward_fill")
        a = result.filter(pl.col("unique_id") == "A")["y"].to_list()
        assert a[1] == 1.0  # filled from previous
        assert a[3] == 3.0

    def test_backward_fill(self):
        result = impute(_make_df_with_nulls(), method="backward_fill")
        a = result.filter(pl.col("unique_id") == "A")["y"].to_list()
        assert a[1] == 3.0  # filled from next

    def test_linear(self):
        result = impute(_make_df_with_nulls(), method="linear")
        a = result.filter(pl.col("unique_id") == "A")["y"].to_list()
        assert a[1] == pytest.approx(2.0)  # interpolated between 1 and 3

    def test_mean(self):
        result = impute(_make_df_with_nulls(), method="mean")
        a = result.filter(pl.col("unique_id") == "A")["y"].to_list()
        group_mean = (1.0 + 3.0 + 5.0 + 6.0) / 4
        assert a[1] == pytest.approx(group_mean)

    def test_median(self):
        result = impute(_make_df_with_nulls(), method="median")
        assert result["y"].null_count() == 0

    def test_seasonal(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8,
                "ds": [date(2024, 1, i + 1) for i in range(8)],
                "y": [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0],
            }
        )
        result = impute(df, method="seasonal", season_length=4)
        assert result["y"][4] == pytest.approx(1.0)  # same position in previous season

    def test_seasonal_requires_season_length(self):
        with pytest.raises(ValueError, match="season_length"):
            impute(_make_df_with_nulls(), method="seasonal")

    def test_add_indicator(self):
        result = impute(_make_df_with_nulls(), method="forward_fill", add_indicator=True)
        assert "y_imputed" in result.columns
        a = result.filter(pl.col("unique_id") == "A")["y_imputed"].to_list()
        assert a[0] is False or a[0] == False  # noqa: E712
        assert a[1] is True or a[1] == True  # noqa: E712

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            impute(_make_df_with_nulls(), method="invalid")

    def test_no_nulls_unchanged(self):
        df = pl.DataFrame(
            {"unique_id": ["A"] * 3, "ds": [date(2024, 1, i + 1) for i in range(3)], "y": [1.0, 2.0, 3.0]}
        )
        result = impute(df, method="forward_fill")
        assert result["y"].to_list() == [1.0, 2.0, 3.0]

    def test_group_independence(self):
        result = impute(_make_df_with_nulls(), method="mean")
        a_mean = result.filter(pl.col("unique_id") == "A")["y"][1]
        b_mean = result.filter(pl.col("unique_id") == "B")["y"][1]
        assert a_mean != b_mean  # Different groups have different means


def test_top_level_import():
    import polars_ts

    assert polars_ts.impute is impute
