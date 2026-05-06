"""Tests for native deep learning forecasters (#150).

Uses lightweight mocks and tiny models to avoid GPU requirements.
Tests marked with ``pytest.mark.skipif`` for torch availability.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")


def _make_panel_df(n_series: int = 3, n_obs: int = 100) -> pl.DataFrame:
    """Create panel DataFrame with trend + noise."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    base = date(2024, 1, 1)
    for sid in [chr(ord("A") + i) for i in range(n_series)]:
        for t in range(n_obs):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(days=t),
                    "y": float(50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1)),
                }
            )
    return pl.DataFrame(rows)


# ── N-BEATS tests ────────────────────────────────────────────────────────


class TestNBEATS:
    def test_fit_predict(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        df = _make_panel_df(n_series=2, n_obs=80)
        forecaster = NBEATSForecaster(
            h=5,
            input_size=20,
            hidden_size=32,
            n_stacks=2,
            n_blocks=2,
            max_epochs=2,
            batch_size=16,
        )
        forecaster.fit(df)
        assert forecaster.is_fitted_

        result = forecaster.predict(df)
        assert isinstance(result, pl.DataFrame)
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y_hat" in result.columns
        assert len(result) == 2 * 5  # n_series * h

    def test_output_future_dates(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster = NBEATSForecaster(h=3, input_size=10, max_epochs=1)
        forecaster.fit(df)
        result = forecaster.predict(df)

        last_date = df["ds"].max()
        first_forecast = result["ds"].min()
        assert first_forecast > last_date

    def test_predict_before_fit_raises(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        forecaster = NBEATSForecaster(h=5)
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict(_make_panel_df())

    def test_custom_columns(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        df = pl.DataFrame(
            {
                "sid": ["X"] * 50,
                "t": [date(2024, 1, 1) + timedelta(days=i) for i in range(50)],
                "val": [float(i) for i in range(50)],
            }
        )
        forecaster = NBEATSForecaster(
            h=3,
            input_size=10,
            max_epochs=1,
            id_col="sid",
            time_col="t",
            target_col="val",
        )
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert "sid" in result.columns
        assert "t" in result.columns
        assert len(result) == 3

    def test_interpretable_mode(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        forecaster = NBEATSForecaster(
            h=5,
            input_size=15,
            stack_types=["trend", "seasonality"],
            max_epochs=1,
        )
        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == 5

    def test_generic_mode(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        forecaster = NBEATSForecaster(
            h=5,
            input_size=15,
            stack_types=["generic", "generic"],
            max_epochs=1,
        )
        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == 5

    def test_multi_series(self):
        from polars_ts.dl.nbeats import NBEATSForecaster

        df = _make_panel_df(n_series=5, n_obs=60)
        forecaster = NBEATSForecaster(h=3, input_size=10, max_epochs=1)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == 5 * 3
        assert set(result["unique_id"].to_list()) == {"A", "B", "C", "D", "E"}


# ── PatchTST tests ──────────────────────────────────────────────────────


class TestPatchTST:
    def test_fit_predict(self):
        from polars_ts.dl.patchtst import PatchTSTForecaster

        df = _make_panel_df(n_series=2, n_obs=80)
        forecaster = PatchTSTForecaster(
            h=5,
            input_size=32,
            patch_len=8,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_epochs=2,
            batch_size=16,
        )
        forecaster.fit(df)
        assert forecaster.is_fitted_

        result = forecaster.predict(df)
        assert isinstance(result, pl.DataFrame)
        assert "y_hat" in result.columns
        assert len(result) == 2 * 5

    def test_predict_before_fit_raises(self):
        from polars_ts.dl.patchtst import PatchTSTForecaster

        forecaster = PatchTSTForecaster(h=5)
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict(_make_panel_df())

    def test_output_schema(self):
        from polars_ts.dl.patchtst import PatchTSTForecaster

        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster = PatchTSTForecaster(
            h=3,
            input_size=20,
            patch_len=5,
            max_epochs=1,
        )
        forecaster.fit(df)
        result = forecaster.predict(df)

        expected_cols = {"unique_id", "ds", "y_hat"}
        assert set(result.columns) == expected_cols
        assert len(result) == 3

    def test_multi_series(self):
        from polars_ts.dl.patchtst import PatchTSTForecaster

        df = _make_panel_df(n_series=4, n_obs=60)
        forecaster = PatchTSTForecaster(
            h=3,
            input_size=20,
            patch_len=5,
            max_epochs=1,
        )
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == 4 * 3


# ── Import tests ─────────────────────────────────────────────────────────


def test_nbeats_importable():
    from polars_ts.dl.nbeats import NBEATSForecaster

    assert NBEATSForecaster is not None


def test_patchtst_importable():
    from polars_ts.dl.patchtst import PatchTSTForecaster

    assert PatchTSTForecaster is not None
