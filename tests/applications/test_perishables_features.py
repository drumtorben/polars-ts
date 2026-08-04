"""Tests for perishables feature engineering (#211)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.applications.perishables import dow_profile, estimate_promo_lift, perishable_calendar_features


@pytest.fixture
def weekly() -> pl.DataFrame:
    """8 weeks of daily data with a strong Saturday spike."""
    start = date(2025, 1, 6)  # a Monday
    n = 56
    dates = [start + timedelta(days=i) for i in range(n)]
    y = [30.0 if d.isoweekday() == 6 else 10.0 for d in dates]
    return pl.DataFrame({"unique_id": ["A"] * n, "ds": dates, "y": y})


class TestCalendarFeatures:
    def test_columns_added(self, weekly):
        out = perishable_calendar_features(weekly)
        for col in ["day_of_week", "month", "is_weekend", "is_month_start", "is_month_end"]:
            assert col in out.columns

    def test_month_boundaries(self):
        df = pl.DataFrame({"ds": [date(2025, 1, 3), date(2025, 1, 15), date(2025, 1, 29), date(2025, 2, 26)]})
        out = perishable_calendar_features(df)
        assert out["is_month_start"].to_list() == [1, 0, 0, 0]
        # Jan 29 is within 5 days of Jan 31; Feb 26 within 5 of Feb 28
        assert out["is_month_end"].to_list() == [0, 0, 1, 1]


class TestDowProfile:
    def test_saturday_uplift(self, weekly):
        prof = dow_profile(weekly)
        assert prof.height == 7
        sat = prof.filter(pl.col("day_of_week") == 6)["dow_index"].item()
        mon = prof.filter(pl.col("day_of_week") == 1)["dow_index"].item()
        # Saturday sells 3x a weekday; index ratio must match
        assert sat / mon == pytest.approx(3.0, rel=1e-6)

    def test_mean_index_is_one(self, weekly):
        prof = dow_profile(weekly)
        assert prof["dow_index"].mean() == pytest.approx(1.0, abs=1e-6)


class TestPromoLift:
    def test_lift_recovered(self):
        rng = np.random.default_rng(3)
        n = 100
        promo = (np.arange(n) % 10 == 0).astype(int)
        base = 10.0 + rng.normal(0, 0.1, n)
        y = np.where(promo == 1, 2.0 * base, base)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * n,
                "ds": [date(2025, 1, 1) + timedelta(days=i) for i in range(n)],
                "y": y,
                "promo": promo,
            }
        )
        out = estimate_promo_lift(df)
        assert out["promo_lift"].item() == pytest.approx(2.0, rel=0.05)
        assert out["n_promo_periods"].item() == 10

    def test_never_promoted_is_null(self):
        df = pl.DataFrame(
            {"unique_id": ["A"] * 3, "ds": [date(2025, 1, i) for i in (1, 2, 3)], "y": [1.0] * 3, "promo": [0] * 3}
        )
        assert estimate_promo_lift(df)["promo_lift"].item() is None

    def test_missing_promo_col_raises(self, weekly):
        with pytest.raises(ValueError, match="ColumnMapping"):
            estimate_promo_lift(weekly)
