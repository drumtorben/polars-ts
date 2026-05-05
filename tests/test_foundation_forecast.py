"""Tests for foundation model forecasting adapters (#151).

Uses lightweight mocks to avoid downloading large models during CI.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


def _make_panel_df(n_series: int = 2, n_obs: int = 50) -> pl.DataFrame:
    """Create a simple panel DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    base = date(2024, 1, 1)
    for sid in [chr(ord("A") + i) for i in range(n_series)]:
        for t in range(n_obs):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(days=t),
                    "y": float(100 + 0.5 * t + rng.normal(0, 1)),
                }
            )
    return pl.DataFrame(rows)


def _mock_chronos_pipeline(h: int, n_samples: int = 10):
    """Create a mock Chronos pipeline instance."""
    _torch = pytest.importorskip("torch")
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = _torch.randn(n_samples, h)
    mock_pipeline.to.return_value = mock_pipeline
    return mock_pipeline


# ── ChronosForecaster tests (mocked) ────────────────────────────────────


class TestChronosForecaster:
    def test_basic_forecast(self):
        pytest.importorskip("torch")
        h = 5
        n_series = 2

        from polars_ts.adapters.foundation_forecast import ChronosForecaster

        forecaster = ChronosForecaster(model_name="amazon/chronos-t5-small")
        forecaster._pipeline = _mock_chronos_pipeline(h)
        result = forecaster.predict(_make_panel_df(n_series=n_series), h=h)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_series * h
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y_hat" in result.columns

    def test_prediction_intervals(self):
        pytest.importorskip("torch")
        h = 3

        from polars_ts.adapters.foundation_forecast import ChronosForecaster

        forecaster = ChronosForecaster(model_name="test/model")
        forecaster._pipeline = _mock_chronos_pipeline(h, n_samples=20)
        result = forecaster.predict(_make_panel_df(n_series=1), h=h)

        assert "y_hat_lower" in result.columns
        assert "y_hat_upper" in result.columns
        assert (result["y_hat_lower"] <= result["y_hat"]).all()
        assert (result["y_hat"] <= result["y_hat_upper"]).all()

    def test_custom_columns(self):
        pytest.importorskip("torch")
        h = 2

        df = pl.DataFrame(
            {
                "sid": ["X"] * 10,
                "t": [date(2024, 1, i + 1) for i in range(10)],
                "val": [float(i) for i in range(10)],
            }
        )

        from polars_ts.adapters.foundation_forecast import ChronosForecaster

        forecaster = ChronosForecaster(
            model_name="test/model",
            id_col="sid",
            time_col="t",
            target_col="val",
        )
        forecaster._pipeline = _mock_chronos_pipeline(h)
        result = forecaster.predict(df, h=h)

        assert "sid" in result.columns
        assert "t" in result.columns
        assert len(result) == h

    def test_future_dates_generated(self):
        pytest.importorskip("torch")
        h = 5
        df = _make_panel_df(n_series=1, n_obs=30)

        from polars_ts.adapters.foundation_forecast import ChronosForecaster

        forecaster = ChronosForecaster(model_name="test/model")
        forecaster._pipeline = _mock_chronos_pipeline(h)
        result = forecaster.predict(df, h=h)

        last_date = df.filter(pl.col("unique_id") == "A")["ds"].max()
        first_forecast_date = result.filter(pl.col("unique_id") == "A")["ds"].min()
        assert first_forecast_date > last_date

    def test_import_error_at_predict(self):
        """ImportError is deferred to predict() time (lazy import)."""
        with patch.dict("sys.modules", {"chronos": None, "torch": None}):
            from polars_ts.adapters.foundation_forecast import ChronosForecaster

            forecaster = ChronosForecaster(model_name="test")
            with pytest.raises(ImportError):
                forecaster.predict(_make_panel_df(n_series=1), h=3)


# ── TimesFMForecaster tests (mocked) ────────────────────────────────────


class TestTimesFMForecaster:
    def _make_mock_timesfm(self, n_series: int, h: int):
        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.random.randn(n_series, h),
            None,
        )
        mock_timesfm = MagicMock()
        mock_timesfm.TimesFm.return_value = mock_model
        return mock_timesfm

    def test_basic_forecast(self):
        h = 5
        n_series = 2
        mock_timesfm = self._make_mock_timesfm(n_series, h)

        with patch.dict("sys.modules", {"timesfm": mock_timesfm}):
            from polars_ts.adapters.foundation_forecast import TimesFMForecaster

            forecaster = TimesFMForecaster()
            result = forecaster.predict(_make_panel_df(n_series=n_series), h=h)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_series * h
        assert "y_hat" in result.columns

    def test_with_quantiles(self):
        h = 3
        n_series = 1

        mock_model = MagicMock()
        mock_model.forecast.return_value = (
            np.random.randn(n_series, h),
            np.random.randn(n_series, h, 2),
        )
        mock_timesfm = MagicMock()
        mock_timesfm.TimesFm.return_value = mock_model

        with patch.dict("sys.modules", {"timesfm": mock_timesfm}):
            from polars_ts.adapters.foundation_forecast import TimesFMForecaster

            forecaster = TimesFMForecaster()
            result = forecaster.predict(_make_panel_df(n_series=n_series), h=h)

        assert "y_hat" in result.columns


# ── MoiraiForecaster tests (mocked) ─────────────────────────────────────


class TestMoiraiForecaster:
    def test_basic_forecast(self):
        _torch = pytest.importorskip("torch")
        h = 5
        n_series = 2
        n_samples = 20

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = (
            _torch.randn(1, n_samples, h),
            _torch.randn(1, h),
        )
        mock_pipeline.to.return_value = mock_pipeline

        mock_moirai_cls = MagicMock()
        mock_moirai_cls.from_pretrained.return_value = mock_pipeline

        mock_uni2ts = MagicMock()
        mock_uni2ts.model.moirai_forecast.MoiraiForecast = mock_moirai_cls

        with patch.dict(
            "sys.modules",
            {
                "uni2ts": mock_uni2ts,
                "uni2ts.model": mock_uni2ts.model,
                "uni2ts.model.moirai_forecast": mock_uni2ts.model.moirai_forecast,
            },
        ):
            from polars_ts.adapters.foundation_forecast import MoiraiForecaster

            forecaster = MoiraiForecaster(model_name="salesforce/moirai-1.1-R-small")
            result = forecaster.predict(_make_panel_df(n_series=n_series), h=h)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_series * h
        assert "y_hat" in result.columns
        assert "y_hat_lower" in result.columns
        assert "y_hat_upper" in result.columns


# ── Unified foundation_forecast() tests ──────────────────────────────────


class TestFoundationForecast:
    def test_chronos_shorthand(self):
        pytest.importorskip("torch")
        h = 3

        from polars_ts.adapters.foundation_forecast import ChronosForecaster, foundation_forecast

        # Patch the class to inject pipeline
        original_load = ChronosForecaster._load_pipeline
        ChronosForecaster._load_pipeline = lambda _self: _mock_chronos_pipeline(h)
        try:
            result = foundation_forecast(
                _make_panel_df(n_series=1),
                model="chronos",
                h=h,
            )
        finally:
            ChronosForecaster._load_pipeline = original_load

        assert isinstance(result, pl.DataFrame)
        assert len(result) == h
        assert "y_hat" in result.columns

    def test_unknown_model_raises(self):
        from polars_ts.adapters.foundation_forecast import foundation_forecast

        with pytest.raises(ValueError, match="Unknown model"):
            foundation_forecast(_make_panel_df(), model="nonexistent", h=5)

    def test_output_schema(self):
        pytest.importorskip("torch")
        h = 3

        from polars_ts.adapters.foundation_forecast import ChronosForecaster, foundation_forecast

        original_load = ChronosForecaster._load_pipeline
        ChronosForecaster._load_pipeline = lambda _self: _mock_chronos_pipeline(h)
        try:
            result = foundation_forecast(
                _make_panel_df(n_series=2),
                model="chronos",
                h=h,
            )
        finally:
            ChronosForecaster._load_pipeline = original_load

        expected_cols = {"unique_id", "ds", "y_hat", "y_hat_lower", "y_hat_upper"}
        assert set(result.columns) == expected_cols


# ── Top-level import tests ───────────────────────────────────────────────


def test_foundation_forecast_importable():
    from polars_ts.adapters.foundation_forecast import foundation_forecast

    assert callable(foundation_forecast)


def test_chronos_forecaster_importable():
    from polars_ts.adapters.foundation_forecast import ChronosForecaster

    assert ChronosForecaster is not None
