"""Tests for LLM-based forecasting adapters (#155).

TDD: tests define the expected API for Time-LLM and LLM-PS adapters.
Uses mocks to avoid downloading large models during CI.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel_df(n_series: int = 2, n_obs: int = 60) -> pl.DataFrame:
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


def _mock_llm_model(h: int):
    """Create a mock LLM model that returns predictions of correct shape."""
    torch = pytest.importorskip("torch")
    mock = MagicMock()
    # forward returns (batch, h) tensor
    mock.return_value = torch.randn(1, h)
    mock.eval.return_value = mock
    mock.to.return_value = mock
    mock.parameters.return_value = iter([torch.zeros(1)])
    return mock


# ---------------------------------------------------------------------------
# TimeLLMForecaster tests
# ---------------------------------------------------------------------------


class TestTimeLLMForecaster:
    def test_basic_forecast(self):
        pytest.importorskip("torch")
        h = 5
        n_series = 2

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        forecaster = TimeLLMForecaster(h=h, input_size=30)
        forecaster.is_fitted_ = True
        forecaster._model = _mock_llm_model(h)
        forecaster._mean = np.zeros(n_series)
        forecaster._std = np.ones(n_series)

        result = forecaster.predict(_make_panel_df(n_series=n_series))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_series * h
        assert "y_hat" in result.columns
        assert "unique_id" in result.columns
        assert "ds" in result.columns

    def test_fit_returns_self(self):
        pytest.importorskip("torch")

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        forecaster = TimeLLMForecaster(h=5, input_size=20, max_epochs=1)
        df = _make_panel_df(n_series=1, n_obs=60)
        result = forecaster.fit(df)
        assert result is forecaster
        assert forecaster.is_fitted_ is True

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch")

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        forecaster = TimeLLMForecaster(h=5, input_size=20)
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict(_make_panel_df())

    def test_custom_columns(self):
        pytest.importorskip("torch")
        h = 3

        df = pl.DataFrame(
            {
                "sid": ["X"] * 40,
                "t": [date(2024, 1, 1) + timedelta(days=i) for i in range(40)],
                "val": [float(i) for i in range(40)],
            }
        )

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        forecaster = TimeLLMForecaster(
            h=h,
            input_size=20,
            max_epochs=1,
            id_col="sid",
            time_col="t",
            target_col="val",
        )
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert "sid" in result.columns
        assert "t" in result.columns
        assert len(result) == h  # 1 series

    def test_output_has_future_dates(self):
        pytest.importorskip("torch")
        h = 5

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster = TimeLLMForecaster(h=h, input_size=20, max_epochs=1)
        forecaster.fit(df)
        result = forecaster.predict(df)

        last_train_date = df.sort("ds")["ds"][-1]
        assert result["ds"].min() > last_train_date

    def test_multi_series(self):
        pytest.importorskip("torch")
        h = 3
        n_series = 3

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        df = _make_panel_df(n_series=n_series, n_obs=60)
        forecaster = TimeLLMForecaster(h=h, input_size=20, max_epochs=1)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == n_series * h

    def test_predictions_are_finite(self):
        pytest.importorskip("torch")
        h = 5

        from polars_ts.adapters.llm_forecast import TimeLLMForecaster

        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster = TimeLLMForecaster(h=h, input_size=20, max_epochs=1)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert np.all(np.isfinite(result["y_hat"].to_numpy()))


# ---------------------------------------------------------------------------
# LLMPSForecaster tests
# ---------------------------------------------------------------------------


class TestLLMPSForecaster:
    def test_basic_forecast(self):
        pytest.importorskip("torch")
        h = 5
        n_series = 2

        from polars_ts.adapters.llm_forecast import LLMPSForecaster

        forecaster = LLMPSForecaster(h=h, input_size=30)
        forecaster.is_fitted_ = True
        forecaster._model = _mock_llm_model(h)
        forecaster._mean = np.zeros(n_series)
        forecaster._std = np.ones(n_series)

        result = forecaster.predict(_make_panel_df(n_series=n_series))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_series * h
        assert "y_hat" in result.columns

    def test_fit_returns_self(self):
        pytest.importorskip("torch")

        from polars_ts.adapters.llm_forecast import LLMPSForecaster

        forecaster = LLMPSForecaster(h=5, input_size=20, max_epochs=1)
        df = _make_panel_df(n_series=1, n_obs=60)
        result = forecaster.fit(df)
        assert result is forecaster
        assert forecaster.is_fitted_ is True

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch")

        from polars_ts.adapters.llm_forecast import LLMPSForecaster

        forecaster = LLMPSForecaster(h=5, input_size=20)
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict(_make_panel_df())

    def test_multi_scale_kernels(self):
        """LLM-PS should use multiple CNN kernel sizes for pattern extraction."""
        pytest.importorskip("torch")

        from polars_ts.adapters.llm_forecast import LLMPSForecaster

        forecaster = LLMPSForecaster(h=5, input_size=30, kernel_sizes=[3, 5, 7])
        df = _make_panel_df(n_series=1, n_obs=60)
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert len(result) == 5
        assert np.all(np.isfinite(result["y_hat"].to_numpy()))

    def test_custom_columns(self):
        pytest.importorskip("torch")
        h = 3

        df = pl.DataFrame(
            {
                "sid": ["X"] * 40,
                "t": [date(2024, 1, 1) + timedelta(days=i) for i in range(40)],
                "val": [float(i) for i in range(40)],
            }
        )

        from polars_ts.adapters.llm_forecast import LLMPSForecaster

        forecaster = LLMPSForecaster(
            h=h,
            input_size=20,
            max_epochs=1,
            id_col="sid",
            time_col="t",
            target_col="val",
        )
        forecaster.fit(df)
        result = forecaster.predict(df)
        assert "sid" in result.columns
        assert len(result) == h


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_timellm_importable_from_adapters(self):
        pytest.importorskip("torch")
        from polars_ts.adapters import TimeLLMForecaster

        assert TimeLLMForecaster is not None

    def test_llmps_importable_from_adapters(self):
        pytest.importorskip("torch")
        from polars_ts.adapters import LLMPSForecaster

        assert LLMPSForecaster is not None

    def test_importable_from_top_level(self):
        pytest.importorskip("torch")
        from polars_ts import LLMPSForecaster, TimeLLMForecaster

        assert TimeLLMForecaster is not None
        assert LLMPSForecaster is not None
