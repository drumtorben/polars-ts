"""CSV ingestion for perishables sales history.

Maps arbitrary source column names onto the canonical polars-ts long
format (``unique_id``, ``ds``, ``y``), validates the schema, aggregates
duplicate (SKU, date) rows, and fills calendar gaps with zero demand —
for perishables a missing day means "no sales", not "missing data".

Raw CSVs live under ``data/perishables/`` which is gitignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ColumnMapping:
    """Maps source CSV column names to canonical polars-ts columns.

    Parameters
    ----------
    sku
        Source column identifying the product (mapped to ``unique_id``).
    date
        Source column with the sale date (mapped to ``ds``).
    quantity
        Source column with units sold (mapped to ``y``).
    category
        Optional product category/family column, kept as ``category``.
        Critical for cold-start borrowing across similar SKUs.
    price
        Optional unit-price column, kept as ``price``.
    promo
        Optional promotion-flag column, kept as ``promo``.

    """

    sku: str = "sku"
    date: str = "date"
    quantity: str = "quantity"
    category: str | None = None
    price: str | None = None
    promo: str | None = None

    def required(self) -> dict[str, str]:
        """Return the mandatory source-to-canonical rename mapping."""
        return {self.sku: "unique_id", self.date: "ds", self.quantity: "y"}

    def optional(self) -> dict[str, str]:
        """Return the optional source-to-canonical rename mapping."""
        out: dict[str, str] = {}
        if self.category is not None:
            out[self.category] = "category"
        if self.price is not None:
            out[self.price] = "price"
        if self.promo is not None:
            out[self.promo] = "promo"
        return out


def validate_sales_frame(df: pl.DataFrame, mapping: ColumnMapping | None = None) -> None:
    """Validate a raw sales frame against a column mapping.

    Parameters
    ----------
    df
        Raw sales DataFrame (source column names).
    mapping
        Column mapping; defaults to ``ColumnMapping()``.

    Raises
    ------
    ValueError
        If mapped columns are missing, dates contain nulls, or the
        quantity column is not numeric.

    """
    mapping = mapping or ColumnMapping()
    expected = list(mapping.required()) + list(mapping.optional())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; available: {df.columns}")
    if df[mapping.date].null_count() > 0:
        raise ValueError(f"Column {mapping.date!r} contains null dates")
    if not df.schema[mapping.quantity].is_numeric():
        raise ValueError(f"Column {mapping.quantity!r} must be numeric, got {df.schema[mapping.quantity]}")


def fill_missing_dates(
    df: pl.DataFrame,
    freq: str = "1d",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pl.DataFrame:
    """Fill calendar gaps per SKU with zero demand.

    Each SKU's calendar runs from its own first sale (its launch) to the
    global last date, so new SKUs are not padded with pre-launch zeros.
    Static columns (``category``) are re-attached; ``promo`` gaps become
    0 and ``price`` gaps are forward/backward filled within each SKU.

    Parameters
    ----------
    df
        Canonical sales frame.
    freq
        Polars interval string for the calendar grid.
    id_col, time_col, target_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Gap-free frame sorted by ``(id_col, time_col)``.

    """
    max_date = df[time_col].max()
    grid = (
        df.group_by(id_col)
        .agg(pl.col(time_col).min().alias("__start"))
        .with_columns(pl.date_ranges("__start", pl.lit(max_date), interval=freq).alias(time_col))
        .explode(time_col)
        .drop("__start")
    )
    out = grid.join(df, on=[id_col, time_col], how="left")
    out = out.with_columns(pl.col(target_col).fill_null(0.0))
    if "category" in out.columns:
        out = out.with_columns(pl.col("category").first().over(id_col))
    if "promo" in out.columns:
        out = out.with_columns(pl.col("promo").fill_null(0))
    if "price" in out.columns:
        out = out.with_columns(pl.col("price").forward_fill().backward_fill().over(id_col))
    return out.sort(id_col, time_col)


def load_sales_csv(
    source: str | Path | pl.DataFrame,
    mapping: ColumnMapping | None = None,
    *,
    fill_gaps: bool = True,
    freq: str = "1d",
    clip_negative: bool = True,
) -> pl.DataFrame:
    """Load a perishables sales CSV into canonical long format.

    Parameters
    ----------
    source
        CSV path, or an already-parsed DataFrame (useful for tests and
        multi-file concatenation done upstream).
    mapping
        Source column mapping; defaults to ``ColumnMapping()``.
    fill_gaps
        Fill per-SKU calendar gaps with zero demand.
    freq
        Calendar interval used when ``fill_gaps`` is enabled.
    clip_negative
        Clip negative quantities (returns/corrections) to zero so they
        do not distort demand models.

    Returns
    -------
    pl.DataFrame
        Columns ``[unique_id, ds, y]`` plus any mapped optional columns,
        one row per SKU per period, sorted by ``(unique_id, ds)``.

    """
    mapping = mapping or ColumnMapping()
    df = source if isinstance(source, pl.DataFrame) else pl.read_csv(source, try_parse_dates=True)
    validate_sales_frame(df, mapping)

    rename = {**mapping.required(), **mapping.optional()}
    df = df.select(list(rename)).rename(rename)
    if df.schema["ds"] == pl.String:
        df = df.with_columns(pl.col("ds").str.to_date())
    df = df.with_columns(pl.col("ds").cast(pl.Date), pl.col("unique_id").cast(pl.String), pl.col("y").cast(pl.Float64))
    if clip_negative:
        df = df.with_columns(pl.col("y").clip(lower_bound=0.0))

    # Aggregate duplicate (SKU, date) rows: quantities sum, optional columns keep first
    agg = [pl.col("y").sum()]
    agg += [pl.col(c).first() for c in ("category", "price", "promo") if c in df.columns]
    df = df.group_by("unique_id", "ds").agg(agg)

    if fill_gaps:
        df = fill_missing_dates(df, freq=freq)
    return df.sort("unique_id", "ds")
