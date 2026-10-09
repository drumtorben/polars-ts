"""Tests for growth-aware preprocessing (#211)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.applications.perishables import (
    apply_training_window,
    fit_growth,
    reapply_growth,
    recency_weights,
    remove_growth,
    select_adaptive_window,
)


def _series(uid: str, values: np.ndarray, start: date = date(2024, 1, 1)) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"unique_id": [uid] * len(values), "ds": dates, "y": values.astype(np.float64)})


@pytest.fixture
def growing_and_flat() -> pl.DataFrame:
    n = 365
    t = np.arange(n)
    growing = 10.0 * np.exp(0.003 * t)  # ~3x over the year
    flat = np.full(n, 10.0)
    return pl.concat([_series("grow", growing), _series("flat", flat)])


class TestFitGrowth:
    def test_recovers_growth_rate(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        g = growth.filter(pl.col("unique_id") == "grow")["daily_growth"].item()
        assert g == pytest.approx(0.003, abs=5e-4)

    def test_flat_series_near_zero(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        g = growth.filter(pl.col("unique_id") == "flat")["daily_growth"].item()
        assert abs(g) < 1e-6

    def test_short_history_gets_zero(self):
        df = _series("new", np.array([1.0, 2.0, 3.0]))
        growth = fit_growth(df)
        assert growth["daily_growth"].item() == 0.0


class TestRemoveReapplyGrowth:
    def test_detrended_is_level(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        out = remove_growth(growing_and_flat, growth).filter(pl.col("unique_id") == "grow")
        # After rescaling to today's level, early and late means should match
        early = out.head(90)["y"].mean()
        late = out.tail(90)["y"].mean()
        assert early == pytest.approx(late, rel=0.1)
        assert "y_raw" in out.columns

    def test_reapply_scales_forecast(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        last = growing_and_flat["ds"].max()
        fc = pl.DataFrame(
            {
                "unique_id": ["grow"] * 30,
                "ds": [last + timedelta(days=i + 1) for i in range(30)],
                "y_hat": [1.0] * 30,
            }
        )
        out = reapply_growth(fc, growth)
        g = growth.filter(pl.col("unique_id") == "grow")["daily_growth"].item()
        assert out["y_hat"][0] == pytest.approx(np.exp(g), rel=1e-6)
        assert out["y_hat"][29] == pytest.approx(np.exp(30 * g), rel=1e-6)
        assert out["y_hat"][29] > out["y_hat"][0]


class TestRecencyWeights:
    def test_latest_is_one_and_decays(self, growing_and_flat):
        out = recency_weights(growing_and_flat, half_life=90.0).filter(pl.col("unique_id") == "flat").sort("ds")
        assert out["weight"][-1] == pytest.approx(1.0)
        assert out["weight"][-91] == pytest.approx(0.5, rel=1e-6)
        assert out["weight"].is_sorted()

    def test_invalid_half_life(self, growing_and_flat):
        with pytest.raises(ValueError, match="half_life"):
            recency_weights(growing_and_flat, half_life=0.0)


class TestAdaptiveWindow:
    def test_fast_growth_short_window(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        windows = select_adaptive_window(growth, tolerance=0.25, min_days=56, max_days=730)
        w_grow = windows.filter(pl.col("unique_id") == "grow")["window_days"].item()
        w_flat = windows.filter(pl.col("unique_id") == "flat")["window_days"].item()
        assert w_grow < w_flat
        assert w_flat == 730  # flat SKU keeps max history
        # ln(1.25)/0.003 ~ 74 days
        assert 56 <= w_grow <= 120

    def test_apply_training_window_trims(self, growing_and_flat):
        growth = fit_growth(growing_and_flat)
        windows = select_adaptive_window(growth)
        out = apply_training_window(growing_and_flat, windows)
        n_grow = out.filter(pl.col("unique_id") == "grow").height
        n_flat = out.filter(pl.col("unique_id") == "flat").height
        assert n_grow < n_flat == 365
        w_grow = windows.filter(pl.col("unique_id") == "grow")["window_days"].item()
        assert n_grow == w_grow

    def test_unknown_sku_kept_full(self, growing_and_flat):
        windows = pl.DataFrame({"unique_id": ["grow"], "window_days": [60]})
        out = apply_training_window(growing_and_flat, windows)
        assert out.filter(pl.col("unique_id") == "flat").height == 365
