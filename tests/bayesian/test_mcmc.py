"""Tests for MCMC forecasting wrapper (#119)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.bayesian.mcmc import (
    MCMCForecaster,
    MCMCResult,
    _ar_logpost,
    _local_level_logpost,
    _mh_sample,
    _seasonal_logpost,
    mcmc_forecast,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    values = 10.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


def _make_multi_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    vals_a = 10.0 + np.cumsum(rng.normal(0, 0.5, n))
    vals_b = 20.0 + np.cumsum(rng.normal(0, 0.3, n))
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)] * 2,
            "y": vals_a.tolist() + vals_b.tolist(),
        }
    )


def _make_seasonal_df(n: int = 60, m: int = 12, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 10.0 + 0.1 * t + 3.0 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.5, n)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Log-posterior tests
# ---------------------------------------------------------------------------


class TestLogPosteriors:
    def test_local_level_finite(self):
        y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        params = np.array([1.0, 0.1, 10.0])
        lp = _local_level_logpost(params, y)
        assert np.isfinite(lp)

    def test_local_level_invalid_sigma(self):
        y = np.array([10.0, 11.0])
        params = np.array([-1.0, 0.1, 10.0])
        assert _local_level_logpost(params, y) == -np.inf

    def test_ar_finite(self):
        y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        params = np.array([1.0, 10.0, 0.5])
        lp = _ar_logpost(params, y, p=1)
        assert np.isfinite(lp)

    def test_ar_invalid_sigma(self):
        y = np.array([10.0, 11.0])
        params = np.array([-1.0, 10.0, 0.5])
        assert _ar_logpost(params, y, p=1) == -np.inf

    def test_seasonal_finite(self):
        y = np.array([10.0 + np.sin(2 * np.pi * i / 4) for i in range(20)])
        params = np.array([1.0, 0.1, 0.1, 10.0, 0.0, 1.0, 0.0, -1.0])
        lp = _seasonal_logpost(params, y, season_length=4)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# MH sampler tests
# ---------------------------------------------------------------------------


class TestMHSampler:
    def test_output_shape(self):
        logpost = lambda p: -0.5 * np.sum(p**2)  # noqa: E731
        samples = _mh_sample(logpost, np.array([1.0, 2.0]), n_samples=100, burn_in=50, seed=42)
        assert samples.shape == (100, 2)

    def test_reproducible(self):
        logpost = lambda p: -0.5 * np.sum(p**2)  # noqa: E731
        s1 = _mh_sample(logpost, np.array([1.0]), n_samples=50, burn_in=25, seed=42)
        s2 = _mh_sample(logpost, np.array([1.0]), n_samples=50, burn_in=25, seed=42)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# MCMCForecaster validation
# ---------------------------------------------------------------------------


class TestMCMCForecasterValidation:
    def test_invalid_model(self):
        with pytest.raises(ValueError, match="model"):
            MCMCForecaster(model="invalid")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            MCMCForecaster(backend="stan")

    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            MCMCForecaster(coverage=0.0)

    def test_ar_invalid_p(self):
        with pytest.raises(ValueError, match="p must"):
            MCMCForecaster(model="ar", p=0)

    def test_seasonal_invalid_length(self):
        with pytest.raises(ValueError, match="season_length"):
            MCMCForecaster(model="seasonal", season_length=1)

    def test_predict_before_fit(self):
        est = MCMCForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            est.predict(_make_df(), h=3)

    def test_predict_invalid_horizon(self):
        est = MCMCForecaster(n_samples=50, burn_in=25)
        est.fit(_make_df())
        with pytest.raises(ValueError, match="positive"):
            est.predict(_make_df(), h=0)

    def test_predict_unseen_group(self):
        df_train = _make_df()
        df_pred = pl.DataFrame(
            {
                "unique_id": ["Z"] * 10,
                "ds": [date(2024, 1, 1 + i) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        est = MCMCForecaster(n_samples=50, burn_in=25)
        est.fit(df_train)
        with pytest.raises(ValueError, match="not seen"):
            est.predict(df_pred, h=3)


# ---------------------------------------------------------------------------
# Local level model
# ---------------------------------------------------------------------------


class TestLocalLevel:
    def test_fit_predict_shape(self):
        df = _make_df()
        est = MCMCForecaster(model="local_level", n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=5)
        assert result.columns == ["unique_id", "step", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 5

    def test_credible_intervals(self):
        df = _make_df()
        est = MCMCForecaster(model="local_level", n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=5)
        assert (result["y_hat_lower"] < result["y_hat_upper"]).all()

    def test_multi_group(self):
        df = _make_multi_df()
        est = MCMCForecaster(model="local_level", n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=3)
        assert len(result) == 6  # 2 groups × 3 steps
        assert sorted(result["unique_id"].unique().to_list()) == ["A", "B"]

    def test_fit_returns_self(self):
        est = MCMCForecaster(n_samples=50, burn_in=25)
        returned = est.fit(_make_df())
        assert returned is est
        assert est.is_fitted_

    def test_samples_stored(self):
        df = _make_df()
        est = MCMCForecaster(model="local_level", n_samples=100, burn_in=50)
        est.fit(df)
        result = est._results["A"]
        assert "sigma_obs" in result.samples
        assert "sigma_level" in result.samples
        assert "level0" in result.samples
        assert len(result.samples["sigma_obs"]) == 100


# ---------------------------------------------------------------------------
# AR model
# ---------------------------------------------------------------------------


class TestAR:
    def test_ar1_fit_predict(self):
        df = _make_df()
        est = MCMCForecaster(model="ar", p=1, n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=5)
        assert len(result) == 5
        assert "y_hat" in result.columns

    def test_ar2(self):
        df = _make_df()
        est = MCMCForecaster(model="ar", p=2, n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=3)
        assert len(result) == 3

    def test_ar_samples_keys(self):
        df = _make_df()
        est = MCMCForecaster(model="ar", p=2, n_samples=100, burn_in=50)
        est.fit(df)
        samples = est._results["A"].samples
        assert "sigma" in samples
        assert "mu" in samples
        assert "phi_1" in samples
        assert "phi_2" in samples


# ---------------------------------------------------------------------------
# Seasonal model
# ---------------------------------------------------------------------------


class TestSeasonal:
    def test_fit_predict(self):
        df = _make_seasonal_df(n=60, m=12)
        est = MCMCForecaster(model="seasonal", season_length=12, n_samples=100, burn_in=50)
        est.fit(df)
        result = est.predict(df, h=12)
        assert len(result) == 12

    def test_short_season(self):
        df = _make_seasonal_df(n=24, m=4)
        est = MCMCForecaster(model="seasonal", season_length=4, n_samples=50, burn_in=25)
        est.fit(df)
        result = est.predict(df, h=4)
        assert len(result) == 4

    def test_seasonal_samples_keys(self):
        df = _make_seasonal_df(n=24, m=4)
        est = MCMCForecaster(model="seasonal", season_length=4, n_samples=50, burn_in=25)
        est.fit(df)
        samples = est._results["A"].samples
        assert "sigma_obs" in samples
        assert "sigma_level" in samples
        assert "sigma_season" in samples
        assert "season_0" in samples
        assert "season_3" in samples


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestMCMCForecastFunction:
    def test_local_level(self):
        result = mcmc_forecast(_make_df(), h=3, model="local_level", n_samples=50, burn_in=25)
        assert result.columns == ["unique_id", "step", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 3

    def test_ar(self):
        result = mcmc_forecast(_make_df(), h=3, model="ar", p=1, n_samples=50, burn_in=25)
        assert len(result) == 3

    def test_seasonal(self):
        result = mcmc_forecast(
            _make_seasonal_df(n=24, m=4), h=4, model="seasonal", season_length=4, n_samples=50, burn_in=25
        )
        assert len(result) == 4

    def test_multi_group(self):
        result = mcmc_forecast(_make_multi_df(), h=3, model="local_level", n_samples=50, burn_in=25)
        assert len(result) == 6

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "sid": ["X"] * 30,
                "ts": [date(2024, 1, 1 + i) for i in range(30)],
                "val": [float(i) for i in range(30)],
            }
        )
        result = mcmc_forecast(
            df, h=3, model="local_level", n_samples=50, burn_in=25, id_col="sid", target_col="val", time_col="ts"
        )
        assert "sid" in result.columns
        assert len(result) == 3

    def test_higher_coverage_wider(self):
        df = _make_df()
        r90 = mcmc_forecast(df, h=3, coverage=0.9, n_samples=200, burn_in=100)
        r50 = mcmc_forecast(df, h=3, coverage=0.5, n_samples=200, burn_in=100)
        w90 = (r90["y_hat_upper"] - r90["y_hat_lower"]).mean()
        w50 = (r50["y_hat_upper"] - r50["y_hat_lower"]).mean()
        assert w90 > w50


# ---------------------------------------------------------------------------
# MCMCResult dataclass
# ---------------------------------------------------------------------------


class TestMCMCResult:
    def test_fields(self):
        r = MCMCResult(samples={"a": np.array([1.0, 2.0])})
        assert r.forecast is None
        assert len(r.samples["a"]) == 2

    def test_reproducible(self):
        df = _make_df()
        r1 = mcmc_forecast(df, h=3, model="local_level", n_samples=50, burn_in=25, seed=42)
        r2 = mcmc_forecast(df, h=3, model="local_level", n_samples=50, burn_in=25, seed=42)
        assert r1["y_hat"].to_list() == pytest.approx(r2["y_hat"].to_list())


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


def test_top_level_imports():
    from polars_ts.bayesian.mcmc import MCMCForecaster as MF
    from polars_ts.bayesian.mcmc import mcmc_forecast as mf

    assert MF is MCMCForecaster
    assert mf is mcmc_forecast
