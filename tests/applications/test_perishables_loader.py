"""Tests for the perishables CSV loader (#211)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from polars_ts.applications.perishables import ColumnMapping, fill_missing_dates, load_sales_csv, validate_sales_frame


@pytest.fixture
def raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_code": ["A", "A", "A", "B", "B"],
            "sale_date": [date(2025, 1, 1), date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 3), date(2025, 1, 4)],
            "units": [2.0, 3.0, 1.0, -1.0, 4.0],
            "family": ["dairy", "dairy", "dairy", "bakery", "bakery"],
        }
    )


@pytest.fixture
def mapping() -> ColumnMapping:
    return ColumnMapping(sku="product_code", date="sale_date", quantity="units", category="family")


class TestValidate:
    def test_missing_column_raises(self, raw):
        with pytest.raises(ValueError, match="Missing columns"):
            validate_sales_frame(raw, ColumnMapping(sku="nope", date="sale_date", quantity="units"))

    def test_null_date_raises(self, mapping):
        df = pl.DataFrame({"product_code": ["A"], "sale_date": [None], "units": [1.0], "family": ["dairy"]})
        with pytest.raises(ValueError, match="null dates"):
            validate_sales_frame(df, mapping)

    def test_non_numeric_quantity_raises(self, mapping):
        df = pl.DataFrame(
            {"product_code": ["A"], "sale_date": [date(2025, 1, 1)], "units": ["two"], "family": ["dairy"]}
        )
        with pytest.raises(ValueError, match="must be numeric"):
            validate_sales_frame(df, mapping)


class TestLoad:
    def test_canonical_columns_and_types(self, raw, mapping):
        out = load_sales_csv(raw, mapping)
        assert out.columns[:3] == ["unique_id", "ds", "y"]
        assert "category" in out.columns
        assert out.schema["ds"] == pl.Date
        assert out.schema["y"] == pl.Float64

    def test_duplicates_summed(self, raw, mapping):
        out = load_sales_csv(raw, mapping, fill_gaps=False)
        a_first = out.filter((pl.col("unique_id") == "A") & (pl.col("ds") == date(2025, 1, 1)))
        assert a_first["y"].item() == 5.0

    def test_negative_clipped(self, raw, mapping):
        out = load_sales_csv(raw, mapping, fill_gaps=False)
        b_neg = out.filter((pl.col("unique_id") == "B") & (pl.col("ds") == date(2025, 1, 3)))
        assert b_neg["y"].item() == 0.0

    def test_csv_roundtrip(self, raw, mapping, tmp_path):
        path = tmp_path / "sales.csv"
        raw.write_csv(path)
        out = load_sales_csv(path, mapping)
        assert out.equals(load_sales_csv(raw, mapping))


class TestFillMissingDates:
    def test_gaps_filled_with_zero(self, raw, mapping):
        out = load_sales_csv(raw, mapping)
        a = out.filter(pl.col("unique_id") == "A")
        # A runs from its launch (Jan 1) to the global max (Jan 4)
        assert a.height == 4
        assert a.filter(pl.col("ds") == date(2025, 1, 2))["y"].item() == 0.0

    def test_no_prelaunch_padding(self, raw, mapping):
        out = load_sales_csv(raw, mapping)
        b = out.filter(pl.col("unique_id") == "B")
        assert b["ds"].min() == date(2025, 1, 3)

    def test_static_category_propagated(self, raw, mapping):
        out = load_sales_csv(raw, mapping)
        assert out.filter(pl.col("unique_id") == "A")["category"].unique().to_list() == ["dairy"]

    def test_promo_gap_zero_filled(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": [date(2025, 1, 1), date(2025, 1, 3)],
                "y": [1.0, 2.0],
                "promo": [1, 1],
            }
        )
        out = fill_missing_dates(df)
        assert out.filter(pl.col("ds") == date(2025, 1, 2))["promo"].item() == 0
