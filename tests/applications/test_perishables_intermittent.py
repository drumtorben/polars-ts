"""Tests for intermittent-demand models (#211)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.applications.perishables import (
    classify_demand,
    croston_forecast,
    intermittent_forecast,
    sba_forecast,
    tsb_forecast,
)


def _series(uid: str, values: list[float]) -> pl.DataFrame:
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"unique_id": [uid] * len(values), "ds": dates, "y": [float(v) for v in values]})


@pytest.fixture
def sparse() -> pl.DataFrame:
    # Demand of ~2 units every 4th day: rate = 0.5/day
    values = [0.0, 0.0, 0.0, 2.0] * 15
    return _series("sparse", values)


@pytest.fixture
def mixed(sparse) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    smooth = _series("smooth", list(10.0 + rng.normal(0, 0.5, 60)))
    return pl.concat([sparse, smooth])


class TestClassifyDemand:
    def test_quadrants(self, mixed):
        out = classify_demand(mixed)
        assert out.filter(pl.col("unique_id") == "sparse")["demand_class"].item() == "intermittent"
        assert out.filter(pl.col("unique_id") == "smooth")["demand_class"].item() == "smooth"

    def test_adi_value(self, sparse):
        out = classify_demand(sparse)
        assert out["adi"].item() == pytest.approx(4.0)

    def test_all_zero_sku(self):
        out = classify_demand(_series("dead", [0.0] * 30))
        assert np.isinf(out["adi"].item())
        assert out["demand_class"].item() == "intermittent"


class TestCroston:
    def test_regular_pattern_rate(self, sparse):
        fc = croston_forecast(sparse, h=7, alpha=0.2)
        assert fc.height == 7
        assert fc["y_hat"].unique().len() == 1  # flat forecast
        assert fc["y_hat"][0] == pytest.approx(0.5, rel=0.05)

    def test_future_dates_follow_history(self, sparse):
        fc = croston_forecast(sparse, h=3)
        assert fc["ds"].min() == sparse["ds"].max() + timedelta(days=1)

    def test_all_zero_forecasts_zero(self):
        fc = croston_forecast(_series("dead", [0.0] * 30), h=5)
        assert fc["y_hat"].to_list() == [0.0] * 5

    def test_invalid_params(self, sparse):
        with pytest.raises(ValueError, match="alpha"):
            croston_forecast(sparse, h=5, alpha=1.5)
        with pytest.raises(ValueError, match="Horizon"):
            croston_forecast(sparse, h=0)


class TestSBA:
    def test_bias_correction_below_croston(self, sparse):
        croston = croston_forecast(sparse, h=1, alpha=0.2)["y_hat"].item()
        sba = sba_forecast(sparse, h=1, alpha=0.2)["y_hat"].item()
        assert sba == pytest.approx(croston * 0.9, rel=1e-9)


class TestTSB:
    def test_regular_pattern_rate(self, sparse):
        fc = tsb_forecast(sparse, h=1, alpha=0.2, beta=0.1)
        assert fc["y_hat"].item() == pytest.approx(0.5, rel=0.3)

    def test_obsolescence_decay(self):
        # SKU sells steadily then goes dead for 30 days
        values = [2.0] * 30 + [0.0] * 30
        alive = tsb_forecast(_series("s", [2.0] * 30), h=1, beta=0.1)["y_hat"].item()
        dead = tsb_forecast(_series("s", values), h=1, beta=0.1)["y_hat"].item()
        assert dead < 0.1 * alive


class TestAutoDispatch:
    def test_routes_by_class(self, mixed):
        fc = intermittent_forecast(mixed, h=5, method="auto")
        assert fc.height == 10
        sparse_hat = fc.filter(pl.col("unique_id") == "sparse")["y_hat"][0]
        # sparse SKU routed to SBA (bias-corrected, below plain croston)
        croston_hat = croston_forecast(mixed, h=1).filter(pl.col("unique_id") == "sparse")["y_hat"].item()
        assert sparse_hat < croston_hat

    def test_explicit_methods_match(self, sparse):
        assert intermittent_forecast(sparse, h=2, method="sba").equals(sba_forecast(sparse, h=2))
        assert intermittent_forecast(sparse, h=2, method="tsb").equals(tsb_forecast(sparse, h=2))

    def test_unknown_method(self, sparse):
        with pytest.raises(ValueError, match="Unknown method"):
            intermittent_forecast(sparse, h=2, method="prophet")
