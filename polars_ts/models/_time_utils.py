"""Shared time-series utility functions.

Canonical location for _infer_freq and _make_future_dates,
previously duplicated in baselines.py and arima.py.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def _infer_freq(times: pl.Series) -> timedelta:
    """Infer the time frequency from a sorted datetime/date series."""
    if len(times) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency")
    diffs = times.diff().drop_nulls()
    if diffs.dtype == pl.Duration:
        return diffs.median()  # type: ignore[return-value]
    # Date column -> cast to duration via subtraction
    casted = times.cast(pl.Datetime("ms"))
    diffs = casted.diff().drop_nulls()
    return diffs.median()  # type: ignore[return-value]


def _make_future_dates(last_time: Any, freq: timedelta, h: int) -> list[Any]:
    """Generate h future timestamps starting from last_time + freq."""
    return [last_time + freq * (i + 1) for i in range(h)]
