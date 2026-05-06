"""Tests for Bayesian ETS forecasters (#117)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")

from polars_ts.models.bayesian_ets import (  # noqa: E402
    BayesianETS,
    BayesianETSResult,
    ETSPriors,
    _forecast_from_params,
    _holt_loglik,
    _hw_loglik,
    _log_posterior,
    _map_estimate,
    _pack_params,
    _ses_loglik,
    _unpack_params,
    bayesian_ets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) + 10.0 for i in range(n)] + [float(2 * i) + 5.0 for i in range(n)],
        }
    )


def _make_seasonal(n: int = 48, m: int = 12) -> pl.DataFrame:
    base = date(2024, 1, 1)
    values = [10.0 + 0.5 * i + 5.0 * np.sin(2 * np.pi * i / m) for i in range(n)]
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values,
        }
    )


def _make_multi_seasonal(n: int = 48, m: int = 12) -> pl.DataFrame:
    base = date(2024, 1, 1)
    vals_a = [10.0 + 0.5 * i + 5.0 * np.sin(2 * np.pi * i / m) for i in range(n)]
    vals_b = [20.0 + 0.3 * i + 3.0 * np.cos(2 * np.pi * i / m) for i in range(n)]
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)] * 2,
            "y": vals_a + vals_b,
        }
    )


# ---------------------------------------------------------------------------
# ETSPriors
# ---------------------------------------------------------------------------


class TestETSPriors:
    def test_defaults(self):
        p = ETSPriors()
        assert p.alpha_a == 2.0
        assert p.alpha_b == 2.0
        assert p.level_sigma == 100.0

    def test_custom(self):
        p = ETSPriors(alpha_a=5.0, alpha_b=1.0, level_mu=50.0)
        assert p.alpha_a == 5.0
        assert p.level_mu == 50.0


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    def test_ses_loglik_finite(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        ll = _ses_loglik(values, alpha=0.3, level0=10.0, sigma=1.0)
        assert np.isfinite(ll)

    def test_ses_loglik_better_fit(self):
        values = [10.0, 10.0, 10.0, 10.0]
        ll_good = _ses_loglik(values, alpha=0.3, level0=10.0, sigma=1.0)
        ll_bad = _ses_loglik(values, alpha=0.3, level0=50.0, sigma=1.0)
        assert ll_good > ll_bad

    def test_holt_loglik_finite(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        ll = _holt_loglik(values, 0.3, 0.1, 10.0, 1.0, 1.0)
        assert np.isfinite(ll)

    def test_hw_loglik_additive_finite(self):
        values = [10.0 + 3.0 * np.sin(2 * np.pi * i / 4) for i in range(16)]
        seasons0 = [0.0, 3.0, 0.0, -3.0]
        ll = _hw_loglik(values, 0.3, 0.1, 0.1, 10.0, 0.0, seasons0, 4, True, 1.0)
        assert np.isfinite(ll)

    def test_hw_loglik_multiplicative_finite(self):
        values = [20.0 + 3.0 * np.sin(2 * np.pi * i / 4) for i in range(16)]
        seasons0 = [1.0, 1.15, 1.0, 0.85]
        ll = _hw_loglik(values, 0.3, 0.1, 0.1, 20.0, 0.0, seasons0, 4, False, 1.0)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------


class TestParamPacking:
    def test_ses_roundtrip(self):
        packed = _pack_params("ses", 0.3, None, None, 10.0, None, None, 1.0)
        unpacked = _unpack_params(packed, "ses", 1)
        assert unpacked["alpha"] == pytest.approx(0.3)
        assert unpacked["level0"] == pytest.approx(10.0)
        assert unpacked["sigma"] == pytest.approx(1.0)
        assert unpacked["beta"] is None
        assert unpacked["gamma"] is None

    def test_holt_roundtrip(self):
        packed = _pack_params("holt", 0.3, 0.1, None, 10.0, 1.0, None, 2.0)
        unpacked = _unpack_params(packed, "holt", 1)
        assert unpacked["alpha"] == pytest.approx(0.3)
        assert unpacked["beta"] == pytest.approx(0.1)
        assert unpacked["level0"] == pytest.approx(10.0)
        assert unpacked["trend0"] == pytest.approx(1.0)
        assert unpacked["sigma"] == pytest.approx(2.0)

    def test_hw_roundtrip(self):
        seasons = [1.0, 2.0, 3.0, 4.0]
        packed = _pack_params("holt_winters", 0.3, 0.1, 0.2, 10.0, 0.5, seasons, 1.5)
        unpacked = _unpack_params(packed, "holt_winters", 4)
        assert unpacked["alpha"] == pytest.approx(0.3)
        assert unpacked["beta"] == pytest.approx(0.1)
        assert unpacked["gamma"] == pytest.approx(0.2)
        assert unpacked["level0"] == pytest.approx(10.0)
        assert unpacked["trend0"] == pytest.approx(0.5)
        assert unpacked["seasons0"] == pytest.approx(seasons)
        assert unpacked["sigma"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Log-posterior
# ---------------------------------------------------------------------------


class TestLogPosterior:
    def test_ses_finite(self):
        values = [10.0, 11.0, 12.0, 13.0]
        theta = _pack_params("ses", 0.3, None, None, 10.0, None, None, 1.0)
        lp = _log_posterior(theta, values, "ses", 1, True, ETSPriors())
        assert np.isfinite(lp)

    def test_invalid_alpha_returns_neg_inf(self):
        values = [10.0, 11.0, 12.0]
        theta = np.array([1.5, 10.0, 1.0])  # alpha > 1
        lp = _log_posterior(theta, values, "ses", 1, True, ETSPriors())
        assert lp == -np.inf

    def test_invalid_sigma_returns_neg_inf(self):
        values = [10.0, 11.0, 12.0]
        theta = np.array([0.3, 10.0, -1.0])  # sigma < 0
        lp = _log_posterior(theta, values, "ses", 1, True, ETSPriors())
        assert lp == -np.inf


# ---------------------------------------------------------------------------
# MAP estimation
# ---------------------------------------------------------------------------


class TestMAPEstimate:
    def test_ses_map(self):
        values = [10.0 + 0.1 * i for i in range(30)]
        theta = _map_estimate(values, "ses", 1, True, ETSPriors(level_mu=10.0))
        params = _unpack_params(theta, "ses", 1)
        assert 0 < params["alpha"] < 1
        assert params["sigma"] > 0

    def test_holt_map(self):
        values = [10.0 + 0.5 * i for i in range(30)]
        theta = _map_estimate(values, "holt", 1, True, ETSPriors(level_mu=10.0))
        params = _unpack_params(theta, "holt", 1)
        assert 0 < params["alpha"] < 1
        assert 0 < params["beta"] < 1

    def test_hw_map(self):
        m = 4
        values = [10.0 + 3.0 * np.sin(2 * np.pi * i / m) for i in range(24)]
        theta = _map_estimate(values, "holt_winters", m, True, ETSPriors(level_mu=10.0))
        params = _unpack_params(theta, "holt_winters", m)
        assert 0 < params["alpha"] < 1
        assert 0 < params["gamma"] < 1
        assert len(params["seasons0"]) == m


# ---------------------------------------------------------------------------
# Forecasting from parameters
# ---------------------------------------------------------------------------


class TestForecastFromParams:
    def test_ses_flat(self):
        values = [10.0, 11.0, 12.0, 13.0]
        params = {"alpha": 0.3, "level0": 10.0, "sigma": 1.0}
        fc = _forecast_from_params(values, params, "ses", 1, True, 3)
        assert len(fc) == 3
        # SES produces flat forecasts
        assert fc[0] == pytest.approx(fc[1])
        assert fc[1] == pytest.approx(fc[2])

    def test_holt_trend(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        params = {"alpha": 0.5, "beta": 0.3, "level0": 10.0, "trend0": 1.0, "sigma": 1.0}
        fc = _forecast_from_params(values, params, "holt", 1, True, 3)
        assert len(fc) == 3
        # Holt should produce increasing forecasts for trending data
        assert fc[1] > fc[0]
        assert fc[2] > fc[1]

    def test_hw_length(self):
        m = 4
        values = [10.0 + 3.0 * np.sin(2 * np.pi * i / m) for i in range(16)]
        params = {
            "alpha": 0.3,
            "beta": 0.1,
            "gamma": 0.1,
            "level0": 10.0,
            "trend0": 0.0,
            "seasons0": [0.0, 3.0, 0.0, -3.0],
            "sigma": 1.0,
        }
        fc = _forecast_from_params(values, params, "holt_winters", m, True, 8)
        assert len(fc) == 8

    def test_noise_injection(self):
        values = [10.0] * 10
        params = {"alpha": 0.3, "level0": 10.0, "sigma": 1.0}
        rng = np.random.default_rng(42)
        fc1 = _forecast_from_params(values, params, "ses", 1, True, 5, sigma_noise=True, rng=rng)
        # With noise, forecasts should differ from deterministic
        fc_det = _forecast_from_params(values, params, "ses", 1, True, 5)
        assert any(abs(a - b) > 1e-10 for a, b in zip(fc1, fc_det, strict=False))


# ---------------------------------------------------------------------------
# BayesianETS class — validation
# ---------------------------------------------------------------------------


class TestBayesianETSValidation:
    def test_invalid_model(self):
        with pytest.raises(ValueError, match="model"):
            BayesianETS(model="arima")

    def test_invalid_inference(self):
        with pytest.raises(ValueError, match="inference"):
            BayesianETS(inference="vi")

    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            BayesianETS(coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            BayesianETS(coverage=1.0)

    def test_hw_requires_season_length(self):
        with pytest.raises(ValueError, match="season_length"):
            BayesianETS(model="holt_winters", season_length=1)

    def test_invalid_seasonal(self):
        with pytest.raises(ValueError, match="seasonal"):
            BayesianETS(model="holt_winters", season_length=4, seasonal="invalid")

    def test_predict_before_fit(self):
        est = BayesianETS()
        with pytest.raises(RuntimeError, match="fit"):
            est.predict(_make_df(), h=3)

    def test_predict_invalid_horizon(self):
        est = BayesianETS()
        est.fit(_make_df())
        with pytest.raises(ValueError, match="positive"):
            est.predict(_make_df(), h=0)

    def test_holt_short_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [date(2024, 1, 1)],
                "y": [1.0],
            }
        )
        est = BayesianETS(model="holt")
        with pytest.raises(ValueError, match="at least 2"):
            est.fit(df)

    def test_hw_short_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": [date(2024, 1, i + 1) for i in range(5)],
                "y": list(range(5)),
            }
        )
        est = BayesianETS(model="holt_winters", season_length=4)
        with pytest.raises(ValueError, match="2\\*season_length"):
            est.fit(df)

    def test_predict_unseen_group(self):
        df_train = pl.DataFrame(
            {
                "unique_id": ["A"] * 10,
                "ds": [date(2024, 1, i + 1) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        df_pred = pl.DataFrame(
            {
                "unique_id": ["Z"] * 10,
                "ds": [date(2024, 1, i + 1) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        est = BayesianETS()
        est.fit(df_train)
        with pytest.raises(ValueError, match="not seen"):
            est.predict(df_pred, h=3)


# ---------------------------------------------------------------------------
# BayesianETS — MAP inference
# ---------------------------------------------------------------------------


class TestBayesianETSMAP:
    def test_ses_map_output_shape(self):
        df = _make_df()
        est = BayesianETS(model="ses", inference="map")
        est.fit(df)
        result = est.predict(df, h=5)
        assert result.columns == ["unique_id", "ds", "y_hat", "y_hat_lower", "y_hat_upper"]
        # 2 groups × 5 steps
        assert len(result) == 10

    def test_ses_map_credible_intervals(self):
        df = _make_df()
        est = BayesianETS(model="ses", inference="map", coverage=0.9)
        est.fit(df)
        result = est.predict(df, h=5)
        # Lower < y_hat < upper for all rows
        assert (result["y_hat_lower"] < result["y_hat"]).all()
        assert (result["y_hat"] < result["y_hat_upper"]).all()

    def test_ses_map_intervals_widen(self):
        df = _make_df()
        est = BayesianETS(model="ses", inference="map")
        est.fit(df)
        result = est.predict(df, h=5)
        a = result.filter(pl.col("unique_id") == "A")
        widths = (a["y_hat_upper"] - a["y_hat_lower"]).to_list()
        # Intervals should widen with horizon
        for i in range(len(widths) - 1):
            assert widths[i + 1] > widths[i]

    def test_holt_map(self):
        df = _make_df()
        est = BayesianETS(model="holt", inference="map")
        est.fit(df)
        result = est.predict(df, h=3)
        assert len(result) == 6

    def test_hw_additive_map(self):
        df = _make_seasonal()
        est = BayesianETS(
            model="holt_winters",
            inference="map",
            season_length=12,
            seasonal="additive",
        )
        est.fit(df)
        result = est.predict(df, h=12)
        assert len(result) == 12
        assert (result["y_hat_lower"] < result["y_hat_upper"]).all()

    def test_hw_multiplicative_map(self):
        df = _make_seasonal()
        df = df.with_columns((pl.col("y") + 20.0).alias("y"))
        est = BayesianETS(
            model="holt_winters",
            inference="map",
            season_length=12,
            seasonal="multiplicative",
        )
        est.fit(df)
        result = est.predict(df, h=6)
        assert len(result) == 6

    def test_multi_group_map(self):
        df = _make_multi_seasonal()
        est = BayesianETS(
            model="holt_winters",
            inference="map",
            season_length=12,
            seasonal="additive",
        )
        est.fit(df)
        result = est.predict(df, h=6)
        assert len(result) == 12  # 2 groups × 6 steps
        groups = result["unique_id"].unique().to_list()
        assert sorted(groups) == ["A", "B"]

    def test_higher_coverage_wider_intervals(self):
        df = _make_df()
        est_90 = BayesianETS(model="ses", inference="map", coverage=0.9)
        est_90.fit(df)
        r90 = est_90.predict(df, h=3)

        est_50 = BayesianETS(model="ses", inference="map", coverage=0.5)
        est_50.fit(df)
        r50 = est_50.predict(df, h=3)

        w90 = (r90["y_hat_upper"] - r90["y_hat_lower"]).mean()
        w50 = (r50["y_hat_upper"] - r50["y_hat_lower"]).mean()
        assert w90 > w50


# ---------------------------------------------------------------------------
# BayesianETS — MCMC inference
# ---------------------------------------------------------------------------


class TestBayesianETSMCMC:
    def test_ses_mcmc_output_shape(self):
        df = _make_df(n=20)
        est = BayesianETS(
            model="ses",
            inference="mcmc",
            n_samples=100,
            burn_in=50,
        )
        est.fit(df)
        result = est.predict(df, h=3)
        assert result.columns == ["unique_id", "ds", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 6

    def test_ses_mcmc_credible_intervals(self):
        df = _make_df(n=20)
        est = BayesianETS(
            model="ses",
            inference="mcmc",
            n_samples=100,
            burn_in=50,
        )
        est.fit(df)
        result = est.predict(df, h=3)
        assert (result["y_hat_lower"] < result["y_hat_upper"]).all()

    def test_holt_mcmc(self):
        df = _make_df(n=20)
        est = BayesianETS(
            model="holt",
            inference="mcmc",
            n_samples=100,
            burn_in=50,
        )
        est.fit(df)
        result = est.predict(df, h=3)
        assert len(result) == 6

    def test_hw_mcmc(self):
        df = _make_seasonal(n=24, m=4)
        est = BayesianETS(
            model="holt_winters",
            inference="mcmc",
            season_length=4,
            seasonal="additive",
            n_samples=50,
            burn_in=25,
        )
        est.fit(df)
        result = est.predict(df, h=4)
        assert len(result) == 4

    def test_mcmc_posterior_samples_stored(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 20,
                "ds": [date(2024, 1, i + 1) for i in range(20)],
                "y": [float(i) for i in range(20)],
            }
        )
        est = BayesianETS(
            model="ses",
            inference="mcmc",
            n_samples=100,
            burn_in=50,
        )
        est.fit(df)
        result = est._results["A"]
        assert result.posterior_samples is not None
        assert result.posterior_samples.shape == (100, 3)  # alpha, level0, sigma

    def test_mcmc_reproducible(self):
        df = _make_df(n=20)
        est1 = BayesianETS(model="ses", inference="mcmc", n_samples=50, burn_in=25, seed=42)
        est1.fit(df)
        r1 = est1.predict(df, h=3)

        est2 = BayesianETS(model="ses", inference="mcmc", n_samples=50, burn_in=25, seed=42)
        est2.fit(df)
        r2 = est2.predict(df, h=3)

        assert r1["y_hat"].to_list() == pytest.approx(r2["y_hat"].to_list())


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestBayesianETSFunction:
    def test_ses_convenience(self):
        result = bayesian_ets(_make_df(), h=3, model="ses")
        assert result.columns == ["unique_id", "ds", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 6

    def test_holt_convenience(self):
        result = bayesian_ets(_make_df(), h=3, model="holt")
        assert len(result) == 6

    def test_hw_convenience(self):
        result = bayesian_ets(
            _make_seasonal(),
            h=6,
            model="holt_winters",
            season_length=12,
        )
        assert len(result) == 6

    def test_mcmc_convenience(self):
        result = bayesian_ets(
            _make_df(n=20),
            h=3,
            model="ses",
            inference="mcmc",
            n_samples=50,
            burn_in=25,
        )
        assert len(result) == 6

    def test_custom_priors(self):
        priors = ETSPriors(alpha_a=5.0, alpha_b=1.0)
        result = bayesian_ets(_make_df(), h=3, model="ses", priors=priors)
        assert len(result) == 6

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "id": ["X"] * 20,
                "timestamp": [date(2024, 1, i + 1) for i in range(20)],
                "value": [float(i) for i in range(20)],
            }
        )
        result = bayesian_ets(
            df,
            h=3,
            model="ses",
            target_col="value",
            id_col="id",
            time_col="timestamp",
        )
        assert "id" in result.columns
        assert "timestamp" in result.columns
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


def test_top_level_imports():
    import polars_ts

    assert polars_ts.bayesian_ets is bayesian_ets
    assert polars_ts.BayesianETS is BayesianETS
    assert polars_ts.ETSPriors is ETSPriors


def test_models_submodule_imports():
    import polars_ts.models as models

    # bayesian_ets function accessed via getattr (submodule shadows `from` import)
    assert models.bayesian_ets is not None
    assert models.BayesianETS is BayesianETS
    assert models.ETSPriors is ETSPriors


# ---------------------------------------------------------------------------
# BayesianETSResult dataclass
# ---------------------------------------------------------------------------


class TestBayesianETSResult:
    def test_result_fields(self):
        r = BayesianETSResult(map_params={"alpha": 0.3}, posterior_samples=None)
        assert r.map_params == {"alpha": 0.3}
        assert r.posterior_samples is None

    def test_result_with_samples(self):
        samples = np.random.randn(10, 3)
        r = BayesianETSResult(map_params={"alpha": 0.3}, posterior_samples=samples)
        assert r.posterior_samples.shape == (10, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_group(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 20,
                "ds": [date(2024, 1, i + 1) for i in range(20)],
                "y": [float(i) for i in range(20)],
            }
        )
        result = bayesian_ets(df, h=5, model="ses")
        assert len(result) == 5

    def test_constant_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 20,
                "ds": [date(2024, 1, i + 1) for i in range(20)],
                "y": [42.0] * 20,
            }
        )
        result = bayesian_ets(df, h=3, model="ses")
        # Forecast should be close to constant value
        assert all(abs(v - 42.0) < 5.0 for v in result["y_hat"].to_list())

    def test_large_horizon(self):
        df = _make_df(n=30)
        result = bayesian_ets(df, h=50, model="ses")
        assert len(result) == 100  # 2 groups × 50 steps

    def test_fit_returns_self(self):
        est = BayesianETS()
        returned = est.fit(_make_df())
        assert returned is est
        assert est.is_fitted_

    def test_datetime_index(self):
        from datetime import datetime

        n = 20
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * n,
                "ds": [datetime(2024, 1, 1, i, 0) for i in range(n)],
                "y": [float(i) for i in range(n)],
            }
        )
        result = bayesian_ets(df, h=3, model="ses")
        assert len(result) == 3
        assert result["ds"].dtype == pl.Datetime
