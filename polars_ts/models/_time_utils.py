"""Shared time-series utility functions.

Canonical location for _infer_freq and _make_future_dates,
previously duplicated in baselines.py and arima.py.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

import polars as pl


def _infer_freq(times: pl.Series) -> timedelta | int | float:
    """Infer the time frequency from a sorted datetime/date or numeric series."""
    if len(times) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency")
    diffs = times.diff().drop_nulls()
    if diffs.dtype == pl.Duration:
        return diffs.median()  # type: ignore[return-value]
    # Integer/float step index -> plain numeric step
    if times.dtype.is_numeric():
        med = cast("int | float", diffs.median())
        return int(round(med)) if times.dtype.is_integer() else float(med)
    # Date column -> cast to duration via subtraction
    casted = times.cast(pl.Datetime("ms"))
    diffs = casted.diff().drop_nulls()
    return diffs.median()  # type: ignore[return-value]


def _make_future_dates(last_time: Any, freq: timedelta | int | float, h: int) -> list[Any]:
    """Generate h future timestamps starting from last_time + freq."""
    return [last_time + freq * (i + 1) for i in range(h)]
