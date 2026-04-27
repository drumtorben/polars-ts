import numpy as np
import polars as pl
import pytest

from polars_ts.bayesian.bsts import BSTS, BSTSResult, bsts_fit, bsts_forecast


@pytest.fixture
def constant_series():
    """Constant mean + noise."""
    rng = np.random.default_rng(42)
    return 5.0 + rng.normal(0, 0.5, size=100)


@pytest.fixture
def trend_series():
    """Linear trend + noise."""
    rng = np.random.default_rng(42)
    return np.linspace(0, 10, 100) + rng.normal(0, 0.5, size=100)


@pytest.fixture
def seasonal_series():
    """Trend + seasonal pattern + noise."""
    rng = np.random.default_rng(42)
    t = np.arange(200)
    return 0.05 * t + 3.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, size=200)


class TestBSTSLocalLevel:
    def test_fit_returns_result(self, constant_series):
        model = BSTS(trend="level")
        result = model.fit(constant_series)
        assert isinstance(result, BSTSResult)

    def test_level_shape(self, constant_series):
        model = BSTS(trend="level")
        result = model.fit(constant_series)
        assert result.level.shape == (100,)

    def test_no_trend_component(self, constant_series):
        model = BSTS(trend="level")
        result = model.fit(constant_series)
        assert result.trend is None

    def test_level_tracks_constant(self, constant_series):
        model = BSTS(trend="level", sigma_level=0.01, sigma_obs=1.0)
        result = model.fit(constant_series)
        assert abs(result.level[-1] - 5.0) < 1.0


class TestBSTSLocalLinearTrend:
    def test_fit_returns_trend(self, trend_series):
        model = BSTS(trend="local_linear")
        result = model.fit(trend_series)
        assert result.trend is not None
        assert result.trend.shape == (100,)

    def test_trend_is_positive(self, trend_series):
        model = BSTS(trend="local_linear", sigma_level=0.1, sigma_trend=0.01)
        result = model.fit(trend_series)
        assert result.trend is not None
        assert result.trend[-1] > 0

    def test_level_shape(self, trend_series):
        model = BSTS(trend="local_linear")
        result = model.fit(trend_series)
        assert result.level.shape == (100,)


class TestBSTSSeasonal:
    def test_seasonal_component(self, seasonal_series):
        model = BSTS(trend="local_linear", seasonal=12)
        result = model.fit(seasonal_series)
        assert result.seasonal is not None
        assert result.seasonal.shape == (200,)

    def test_seasonal_has_periodicity(self, seasonal_series):
        model = BSTS(trend="local_linear", seasonal=12, sigma_seasonal=0.1)
        result = model.fit(seasonal_series)
        assert result.seasonal is not None
        # Seasonal component should have non-trivial variance
        assert np.std(result.seasonal) > 0.1

    def test_no_seasonal_when_none(self, trend_series):
        model = BSTS(trend="local_linear", seasonal=None)
        result = model.fit(trend_series)
        assert result.seasonal is None


class TestBSTSForecast:
    def test_forecast_shape(self, trend_series):
        model = BSTS(trend="local_linear")
        result = model.forecast(trend_series, h=12)
        assert result.forecast is not None
        assert result.forecast.shape == (12,)

    def test_forecast_var_shape(self, trend_series):
        model = BSTS(trend="local_linear")
        result = model.forecast(trend_series, h=12)
        assert result.forecast_var is not None
        assert result.forecast_var.shape == (12,)

    def test_forecast_var_positive(self, trend_series):
        model = BSTS(trend="local_linear")
        result = model.forecast(trend_series, h=12)
        assert result.forecast_var is not None
        assert np.all(result.forecast_var > 0)

    def test_forecast_var_increases(self, trend_series):
        """Forecast uncertainty should grow with horizon."""
        model = BSTS(trend="local_linear")
        result = model.forecast(trend_series, h=20)
        assert result.forecast_var is not None
        assert result.forecast_var[-1] > result.forecast_var[0]

    def test_seasonal_forecast(self, seasonal_series):
        model = BSTS(trend="local_linear", seasonal=12)
        result = model.forecast(seasonal_series, h=24)
        assert result.forecast is not None
        assert result.forecast.shape == (24,)


class TestBSTSEdgeCases:
    def test_unknown_trend_raises(self):
        with pytest.raises(ValueError, match="Unknown trend"):
            model = BSTS(trend="quadratic")
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_missing_values(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        model = BSTS(trend="level")
        result = model.fit(y)
        assert np.all(np.isfinite(result.level))


class TestBSTSFunctions:
    def test_bsts_fit_panel(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 50 + ["B"] * 50,
                "y": (5.0 + rng.normal(0, 0.5, 50)).tolist() + (10.0 + rng.normal(0, 0.5, 50)).tolist(),
            }
        )
        results = bsts_fit(df, trend="level")
        assert set(results.keys()) == {"A", "B"}
        assert results["A"].level.shape == (50,)

    def test_bsts_forecast_panel(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 50,
                "y": (np.linspace(0, 10, 50) + rng.normal(0, 0.5, 50)).tolist(),
            }
        )
        results = bsts_forecast(df, h=12, trend="local_linear")
        assert results["A"].forecast is not None
        assert results["A"].forecast.shape == (12,)

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 20, "val": [float(i) for i in range(20)]})
        results = bsts_fit(df, trend="level", id_col="sid", target_col="val")
        assert "X" in results


class TestBSTSImports:
    def test_top_level_import(self):
        from polars_ts import BSTS as B

        assert B is BSTS

    def test_functional_imports(self):
        from polars_ts import bsts_fit as bf
        from polars_ts import bsts_forecast as bfc

        assert callable(bf)
        assert callable(bfc)

    def test_submodule_import(self):
        from polars_ts.bayesian import BSTS as B

        assert B is BSTS
