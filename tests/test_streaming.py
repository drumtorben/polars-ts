"""Tests for streaming / online learning module (issue #164)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_series():
    """Two daily time series with 20 observations each."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 20 + ["B"] * 20,
            "ds": dates * 2,
            "y": [float(i) + np.sin(i) for i in range(20)] + [float(i) * 0.5 + np.cos(i) for i in range(20)],
        }
    )


@pytest.fixture
def new_observations():
    """Create new batch of observations for partial_fit."""
    dates = [date(2024, 1, 21) + timedelta(days=i) for i in range(5)]
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 5 + ["B"] * 5,
            "ds": dates * 2,
            "y": [20.0 + np.sin(20 + i) for i in range(5)] + [10.0 + np.cos(20 + i) for i in range(5)],
        }
    )


# ===========================================================================
# StreamingETS
# ===========================================================================


class TestStreamingETS:
    """Test online exponential smoothing."""

    def test_fit_returns_self(self, daily_series):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        result = model.fit(daily_series)
        assert result is model

    def test_is_fitted_flag(self, daily_series):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        assert not model.is_fitted_
        model.fit(daily_series)
        assert model.is_fitted_

    def test_predict_before_fit_raises(self):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(h=3)

    def test_partial_fit_before_fit_raises(self, new_observations):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        with pytest.raises(RuntimeError, match="fit"):
            model.partial_fit(new_observations)

    def test_partial_fit_returns_self(self, daily_series, new_observations):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        model.fit(daily_series)
        result = model.partial_fit(new_observations)
        assert result is model

    def test_predict_output_schema(self, daily_series):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        model.fit(daily_series)
        forecast = model.predict(h=3)
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "y_hat" in forecast.columns
        assert forecast.shape[0] == 6  # 2 series * 3 steps

    def test_predict_after_partial_fit_uses_updated_state(self, daily_series, new_observations):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        model.fit(daily_series)
        forecast_before = model.predict(h=1)

        model.partial_fit(new_observations)
        forecast_after = model.predict(h=1)

        # Forecasts should differ since state was updated
        assert not forecast_before["y_hat"].to_list() == pytest.approx(forecast_after["y_hat"].to_list())

    def test_holt_method(self, daily_series):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="holt", alpha=0.3, beta=0.1)
        model.fit(daily_series)
        forecast = model.predict(h=3)
        assert forecast.shape[0] == 6
        # Holt produces trending forecasts (not flat)
        a_forecasts = forecast.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a_forecasts[0] != a_forecasts[2]

    def test_holt_winters_method(self):
        """Holt-Winters requires enough data for two full seasons."""
        from polars_ts.streaming import StreamingETS

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        seasonal_vals = [10.0 + 5 * np.sin(2 * np.pi * i / 7) + i * 0.1 for i in range(30)]
        df = pl.DataFrame({"unique_id": ["A"] * 30, "ds": dates, "y": seasonal_vals})
        model = StreamingETS(method="holt_winters", alpha=0.3, beta=0.1, gamma=0.1, season_length=7)
        model.fit(df)
        forecast = model.predict(h=7)
        assert forecast.shape[0] == 7

    def test_state_property(self, daily_series):
        """Expose internal state for inspection."""
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        model.fit(daily_series)
        state = model.state_
        assert "A" in state
        assert "B" in state
        assert "level" in state["A"]

    def test_n_seen_tracks_observations(self, daily_series, new_observations):
        from polars_ts.streaming import StreamingETS

        model = StreamingETS(method="ses", alpha=0.3)
        model.fit(daily_series)
        assert model.state_["A"]["n_obs"] == 20

        model.partial_fit(new_observations)
        assert model.state_["A"]["n_obs"] == 25


# ===========================================================================
# StreamingKalmanFilter
# ===========================================================================


class TestStreamingKalmanFilter:
    """Test online Kalman filter with single-observation update."""

    def test_init_and_fit(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            Q=np.eye(2) * 0.01,
            R=np.array([[1.0]]),
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kf.fit(y)
        assert kf.is_fitted_

    def test_update_single_observation(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            Q=np.eye(2) * 0.01,
            R=np.array([[1.0]]),
        )
        y = np.array([1.0, 2.0, 3.0])
        kf.fit(y)

        state_before = kf.state_mean.copy()
        kf.update(4.0)
        state_after = kf.state_mean

        # State should move toward the new observation
        assert state_after[0] != state_before[0]

    def test_update_returns_self(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
        )
        kf.fit(np.array([1.0, 2.0]))
        result = kf.update(3.0)
        assert result is kf

    def test_predict_ahead(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            Q=np.eye(2) * 0.01,
            R=np.array([[1.0]]),
        )
        kf.fit(np.arange(1.0, 11.0))
        predictions = kf.predict(h=5)

        assert predictions.shape == (5,)
        # For linear trend data, predictions should increase
        assert predictions[-1] > predictions[0]

    def test_update_missing_skips(self):
        """Update with NaN should not crash (skip observation)."""
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
        )
        kf.fit(np.array([1.0, 2.0, 3.0]))
        kf.update(np.nan)
        # State changes only from prediction step, not update
        assert kf.state_mean is not None

    def test_predict_before_fit_raises(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
        )
        with pytest.raises(RuntimeError, match="fit"):
            kf.predict(h=3)

    def test_log_likelihood_increases_with_data(self):
        from polars_ts.streaming import StreamingKalmanFilter

        kf = StreamingKalmanFilter(
            F=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
        )
        kf.fit(np.array([1.0, 1.1, 0.9]))
        ll_before = kf.log_likelihood_

        kf.update(1.0)
        ll_after = kf.log_likelihood_

        # Log-likelihood should change after update
        assert ll_after != ll_before


# ===========================================================================
# SlidingWindowManager
# ===========================================================================


class TestSlidingWindowManager:
    """Test memory-efficient windowed state management."""

    def test_init(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=10)
        assert mgr.window_size == 10

    def test_append_and_get(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        mgr.append("series_A", np.array([1.0, 2.0, 3.0]))
        result = mgr.get("series_A")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_window_eviction(self):
        """Older data beyond window_size should be evicted."""
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        mgr.append("A", np.array([1.0, 2.0, 3.0, 4.0]))
        mgr.append("A", np.array([5.0, 6.0, 7.0]))
        result = mgr.get("A")
        # Should only keep last 5 values
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_multiple_series(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=3)
        mgr.append("A", np.array([1.0, 2.0]))
        mgr.append("B", np.array([10.0, 20.0]))
        np.testing.assert_array_equal(mgr.get("A"), [1.0, 2.0])
        np.testing.assert_array_equal(mgr.get("B"), [10.0, 20.0])

    def test_get_unknown_series_returns_empty(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        result = mgr.get("unknown")
        assert len(result) == 0

    def test_clear(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        mgr.append("A", np.array([1.0, 2.0]))
        mgr.clear("A")
        assert len(mgr.get("A")) == 0

    def test_clear_all(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        mgr.append("A", np.array([1.0]))
        mgr.append("B", np.array([2.0]))
        mgr.clear_all()
        assert len(mgr.get("A")) == 0
        assert len(mgr.get("B")) == 0

    def test_series_ids(self):
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=5)
        mgr.append("A", np.array([1.0]))
        mgr.append("B", np.array([2.0]))
        assert set(mgr.series_ids) == {"A", "B"}

    def test_memory_usage_bounded(self):
        """Memory should not grow beyond window_size per series."""
        from polars_ts.streaming import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=100)
        for i in range(1000):
            mgr.append("A", np.array([float(i)]))
        result = mgr.get("A")
        assert len(result) == 100


# ===========================================================================
# StreamingGlobalForecaster
# ===========================================================================


class TestStreamingGlobalForecaster:
    """Test incremental global model updates."""

    def test_fit_returns_self(self, daily_series):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        result = model.fit(daily_series)
        assert result is model

    def test_is_fitted_flag(self, daily_series):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        assert not model.is_fitted_
        model.fit(daily_series)
        assert model.is_fitted_

    def test_partial_fit_updates_model(self, daily_series, new_observations):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        model.fit(daily_series)
        coef_before = model.estimator_.coef_.copy()

        model.partial_fit(new_observations)
        coef_after = model.estimator_.coef_

        # Coefficients should change after incremental update
        assert not np.allclose(coef_before, coef_after)

    def test_partial_fit_returns_self(self, daily_series, new_observations):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        model.fit(daily_series)
        result = model.partial_fit(new_observations)
        assert result is model

    def test_predict_output_schema(self, daily_series):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        model.fit(daily_series)
        forecast = model.predict(h=3)
        assert "unique_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "y_hat" in forecast.columns
        assert forecast.shape[0] == 6  # 2 series * 3 steps

    def test_predict_before_fit_raises(self):
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=15)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(h=3)

    def test_window_manager_integration(self, daily_series, new_observations):
        """Window manager should keep only recent data."""
        from sklearn.linear_model import SGDRegressor

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=SGDRegressor(), lags=[1, 2, 3], window_size=10)
        model.fit(daily_series)
        model.partial_fit(new_observations)

        # Internal window should be bounded
        for series_id in model.window_manager_.series_ids:
            data = model.window_manager_.get(series_id)
            assert len(data) <= 10

    def test_requires_partial_fit_compatible_estimator(self, daily_series):
        """Estimator must support partial_fit for streaming."""
        from sklearn.linear_model import LinearRegression

        from polars_ts.streaming import StreamingGlobalForecaster

        model = StreamingGlobalForecaster(estimator=LinearRegression(), lags=[1, 2, 3], window_size=15)
        model.fit(daily_series)
        with pytest.raises(TypeError, match="partial_fit"):
            model.partial_fit(daily_series)
