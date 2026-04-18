"""Tests for DL/RL integration adapters (#48)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from polars_ts.adapters.neuralforecast import from_neuralforecast, to_neuralforecast
from polars_ts.adapters.rl_env import ForecastEnv


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 5 + ["B"] * 5,
            "ds": [date(2024, 1, i + 1) for i in range(5)] * 2,
            "y": [float(i) for i in range(10)],
        }
    )


class TestNeuralForecast:
    def test_to_neuralforecast(self):
        result = to_neuralforecast(_make_df())
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_to_neuralforecast_renames(self):
        df = _make_df().rename({"unique_id": "series", "ds": "timestamp", "y": "value"})
        result = to_neuralforecast(df, id_col="series", time_col="timestamp", target_col="value")
        assert "unique_id" in result.columns
        assert "ds" in result.columns
        assert "y" in result.columns

    def test_from_neuralforecast(self):
        nf_out = pl.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": [date(2024, 1, 6), date(2024, 1, 7)],
                "NBEATS": [10.0, 11.0],
            }
        )
        result = from_neuralforecast(nf_out)
        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert result["y_hat"].to_list() == [10.0, 11.0]

    def test_from_neuralforecast_no_forecast_cols(self):
        nf_out = pl.DataFrame({"unique_id": ["A"], "ds": [date(2024, 1, 1)]})
        with pytest.raises(ValueError, match="No forecast columns"):
            from_neuralforecast(nf_out)


class TestPyTorchForecasting:
    def test_to_pytorch_forecasting(self):
        from polars_ts.adapters.pytorch_forecasting import to_pytorch_forecasting

        result = to_pytorch_forecasting(_make_df())
        assert "data" in result
        assert "metadata" in result
        assert result["metadata"]["target"] == "y"
        assert "time_idx" in result["data"].columns

    def test_from_pytorch_forecasting_array(self):
        from polars_ts.adapters.pytorch_forecasting import from_pytorch_forecasting

        preds = np.array([1.0, 2.0, 3.0])
        result = from_pytorch_forecasting(preds)
        assert "y_hat" in result.columns
        assert len(result) == 3


class TestForecastEnv:
    def test_reset(self):
        data = np.arange(20, dtype=np.float64)
        forecasts = np.arange(20, dtype=np.float64) + 0.5
        env = ForecastEnv(data, forecasts, window_size=5)
        obs = env.reset()
        assert len(obs) == 6  # window + forecast

    def test_step(self):
        data = np.arange(20, dtype=np.float64)
        forecasts = np.arange(20, dtype=np.float64) + 0.5
        env = ForecastEnv(data, forecasts, window_size=5)
        env.reset()
        obs, reward, done, info = env.step(5.0)
        assert isinstance(reward, float)
        assert not done
        assert "actual" in info

    def test_episode_terminates(self):
        data = np.arange(15, dtype=np.float64)
        forecasts = np.arange(15, dtype=np.float64)
        env = ForecastEnv(data, forecasts, window_size=5)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(0.0)
            steps += 1
        assert steps == 10  # 15 - 5

    def test_custom_reward(self):
        data = np.ones(20)
        forecasts = np.ones(20)
        env = ForecastEnv(data, forecasts, window_size=5, reward_fn=lambda _a, actual, _fc: actual * 10)
        env.reset()
        _, reward, _, _ = env.step(0.0)
        assert reward == 10.0

    def test_short_data_raises(self):
        with pytest.raises(ValueError, match="longer than"):
            ForecastEnv(np.array([1.0, 2.0]), np.array([1.0, 2.0]), window_size=5)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.to_neuralforecast is to_neuralforecast
    assert polars_ts.ForecastEnv is ForecastEnv
