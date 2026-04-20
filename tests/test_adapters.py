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


# --- Additional NeuralForecast tests ---


def test_neuralforecast_roundtrip():
    """Converting to NF format and back should preserve data."""
    df = _make_df()
    nf_df = to_neuralforecast(df)
    # Simulate NF adding a forecast column
    nf_out = nf_df.with_columns(pl.col("y").alias("NBEATS"))
    result = from_neuralforecast(nf_out)
    assert result["y_hat"].to_list() == df["y"].to_list()


def test_neuralforecast_multi_series_preserved():
    """All series should survive the conversion."""
    df = _make_df()
    result = to_neuralforecast(df)
    assert result["unique_id"].n_unique() == 2
    assert len(result) == 10


def test_neuralforecast_multiple_model_columns():
    """from_neuralforecast should handle multiple model columns (average them)."""
    nf_out = pl.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": [date(2024, 1, 6), date(2024, 1, 7)],
            "NBEATS": [10.0, 12.0],
            "NHITS": [11.0, 13.0],
        }
    )
    result = from_neuralforecast(nf_out)
    assert "y_hat" in result.columns
    # Should average the two model columns
    assert result["y_hat"][0] == pytest.approx(10.5)


# --- Additional PyTorch Forecasting tests ---


def test_pytorch_forecasting_time_idx_monotonic():
    """time_idx should be monotonically increasing per group."""
    from polars_ts.adapters.pytorch_forecasting import to_pytorch_forecasting

    result = to_pytorch_forecasting(_make_df())
    data = result["data"]
    for uid in data["unique_id"].unique().to_list():
        group = data.filter(pl.col("unique_id") == uid)
        time_idx = group["time_idx"].to_list()
        assert time_idx == sorted(time_idx)


def test_pytorch_forecasting_metadata_fields():
    """Metadata should include expected fields."""
    from polars_ts.adapters.pytorch_forecasting import to_pytorch_forecasting

    result = to_pytorch_forecasting(_make_df())
    meta = result["metadata"]
    assert "target" in meta
    assert "group_ids" in meta


# --- Additional ForecastEnv tests ---


def test_forecast_env_observation_shape():
    """Observation should have consistent shape across steps."""
    data = np.arange(20, dtype=np.float64)
    forecasts = np.arange(20, dtype=np.float64) + 0.5
    env = ForecastEnv(data, forecasts, window_size=5)
    obs1 = env.reset()
    obs2, _, _, _ = env.step(5.0)
    assert len(obs1) == len(obs2)


def test_forecast_env_reward_negative_for_bad_action():
    """A far-off action should produce worse reward than a close one."""
    data = np.ones(20) * 10.0
    forecasts = np.ones(20) * 10.0
    env = ForecastEnv(data, forecasts, window_size=5)
    env.reset()
    _, reward_good, _, _ = env.step(10.0)  # perfect action
    env.reset()
    _, reward_bad, _, _ = env.step(100.0)  # terrible action
    assert reward_good >= reward_bad


def test_top_level_imports():
    import polars_ts

    assert polars_ts.to_neuralforecast is to_neuralforecast
    assert polars_ts.ForecastEnv is ForecastEnv
