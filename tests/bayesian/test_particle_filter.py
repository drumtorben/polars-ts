"""Tests for Particle Filter / Sequential Monte Carlo (#122)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.bayesian.particle_filter import (
    ParticleFilter,
    ParticleFilterResult,
    _systematic_resample,
    ar1_transition,
    local_level_loglik,
    local_level_transition,
    particle_filter,
    stochastic_volatility_loglik,
    stochastic_volatility_transition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_local_level_data(n: int = 100, sigma_level: float = 0.5, sigma_obs: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    states = np.zeros(n)
    obs = np.zeros(n)
    states[0] = rng.normal(0, sigma_level)
    obs[0] = states[0] + rng.normal(0, sigma_obs)
    for t in range(1, n):
        states[t] = states[t - 1] + rng.normal(0, sigma_level)
        obs[t] = states[t] + rng.normal(0, sigma_obs)
    return states, obs


def _make_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    _, obs = _make_local_level_data(n, seed=seed)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": obs.tolist(),
        }
    )


def _make_multi_df(n: int = 30, seed: int = 42) -> pl.DataFrame:
    _, obs_a = _make_local_level_data(n, seed=seed)
    _, obs_b = _make_local_level_data(n, seed=seed + 1)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)] * 2,
            "y": obs_a.tolist() + obs_b.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Resampling tests
# ---------------------------------------------------------------------------


class TestResampling:
    def test_systematic_output_shape(self):
        rng = np.random.default_rng(42)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        indices = _systematic_resample(weights, rng)
        assert len(indices) == 4
        assert all(0 <= i < 4 for i in indices)

    def test_systematic_concentrates_on_heavy(self):
        rng = np.random.default_rng(42)
        weights = np.array([0.01, 0.01, 0.01, 0.97])
        indices = _systematic_resample(weights, rng)
        assert np.sum(indices == 3) >= 3

    def test_uniform_weights(self):
        rng = np.random.default_rng(42)
        weights = np.ones(100) / 100
        indices = _systematic_resample(weights, rng)
        assert len(indices) == 100
        assert len(np.unique(indices)) > 50


# ---------------------------------------------------------------------------
# Built-in model functions
# ---------------------------------------------------------------------------


class TestBuiltinModels:
    def test_local_level_transition(self):
        rng = np.random.default_rng(42)
        fn = local_level_transition(sigma_level=0.5)
        particles = np.zeros(100)
        new = fn(particles, 1, rng)
        assert new.shape == (100,)
        assert np.std(new) > 0

    def test_local_level_loglik(self):
        fn = local_level_loglik(sigma_obs=1.0)
        particles = np.array([0.0, 1.0, 2.0, 10.0])
        log_w = fn(particles, 0.0)
        assert log_w.shape == (4,)
        assert log_w[0] > log_w[3]  # particle at 0 closer to obs=0

    def test_ar1_transition(self):
        rng = np.random.default_rng(42)
        fn = ar1_transition(phi=0.9, sigma=0.1, mu=0.0)
        particles = np.ones(100) * 5.0
        new = fn(particles, 1, rng)
        assert np.mean(new) == pytest.approx(4.5, abs=0.5)

    def test_sv_transition(self):
        rng = np.random.default_rng(42)
        fn = stochastic_volatility_transition(phi=0.95, sigma_v=0.2)
        particles = np.zeros(100)
        new = fn(particles, 1, rng)
        assert new.shape == (100,)

    def test_sv_loglik(self):
        fn = stochastic_volatility_loglik()
        particles = np.array([0.0, 1.0, 2.0])
        log_w = fn(particles, 1.0)
        assert log_w.shape == (3,)
        assert all(np.isfinite(log_w))


# ---------------------------------------------------------------------------
# ParticleFilter validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_few_particles(self):
        with pytest.raises(ValueError, match="n_particles"):
            ParticleFilter(n_particles=1)

    def test_invalid_resample_method(self):
        with pytest.raises(ValueError, match="resample_method"):
            ParticleFilter(resample_method="stratified")

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="resample_threshold"):
            ParticleFilter(resample_threshold=0.0)

    def test_missing_transition(self):
        pf = ParticleFilter(observation_loglik=local_level_loglik())
        with pytest.raises(ValueError, match="transition_fn"):
            pf.filter(np.array([1.0, 2.0]))

    def test_missing_loglik(self):
        pf = ParticleFilter(transition_fn=local_level_transition())
        with pytest.raises(ValueError, match="observation_loglik"):
            pf.filter(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Particle filter on local level model
# ---------------------------------------------------------------------------


class TestLocalLevelFilter:
    def test_output_shape(self):
        _, obs = _make_local_level_data(50)
        pf = ParticleFilter(
            n_particles=500,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs)
        assert result.filtered_mean.shape == (50,)
        assert result.filtered_var.shape == (50,)
        assert result.ess.shape == (50,)

    def test_tracks_true_state(self):
        states, obs = _make_local_level_data(100, sigma_level=0.5, sigma_obs=1.0)
        pf = ParticleFilter(
            n_particles=1000,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs)
        rmse = np.sqrt(np.mean((result.filtered_mean - states) ** 2))
        assert rmse < 2.0  # should track reasonably well

    def test_ess_positive(self):
        _, obs = _make_local_level_data(50)
        pf = ParticleFilter(
            n_particles=500,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs)
        assert np.all(result.ess > 0)

    def test_log_likelihood_finite(self):
        _, obs = _make_local_level_data(50)
        pf = ParticleFilter(
            n_particles=500,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs)
        assert np.isfinite(result.log_likelihood)

    def test_store_history(self):
        _, obs = _make_local_level_data(20)
        pf = ParticleFilter(
            n_particles=100,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            store_history=True,
        )
        result = pf.filter(obs)
        assert result.particles_history is not None
        assert result.particles_history.shape == (20, 100)
        assert result.weights_history is not None
        assert result.weights_history.shape == (20, 100)

    def test_no_history_by_default(self):
        _, obs = _make_local_level_data(20)
        pf = ParticleFilter(
            n_particles=100,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs)
        assert result.particles_history is None
        assert result.weights_history is None

    def test_multinomial_resampling(self):
        _, obs = _make_local_level_data(30)
        pf = ParticleFilter(
            n_particles=500,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            resample_method="multinomial",
        )
        result = pf.filter(obs)
        assert result.filtered_mean.shape == (30,)

    def test_reproducible(self):
        _, obs = _make_local_level_data(30)
        pf1 = ParticleFilter(
            n_particles=200,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            seed=42,
        )
        pf2 = ParticleFilter(
            n_particles=200,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            seed=42,
        )
        r1 = pf1.filter(obs)
        r2 = pf2.filter(obs)
        np.testing.assert_array_equal(r1.filtered_mean, r2.filtered_mean)

    def test_more_particles_lower_variance(self):
        _, obs = _make_local_level_data(30)
        pf_small = ParticleFilter(
            n_particles=50,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            seed=42,
        )
        pf_large = ParticleFilter(
            n_particles=2000,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            seed=42,
        )
        r_small = pf_small.filter(obs)
        r_large = pf_large.filter(obs)
        assert np.mean(r_large.filtered_var) <= np.mean(r_small.filtered_var) * 2

    def test_initial_particles(self):
        _, obs = _make_local_level_data(20)
        init = np.ones(200) * obs[0]
        pf = ParticleFilter(
            n_particles=200,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
        )
        result = pf.filter(obs, initial_particles=init)
        assert result.filtered_mean.shape == (20,)


# ---------------------------------------------------------------------------
# Stochastic volatility model
# ---------------------------------------------------------------------------


class TestStochasticVolatility:
    def test_sv_filter(self):
        rng = np.random.default_rng(42)
        n = 50
        h = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            h[t] = 0.95 * h[t - 1] + rng.normal(0, 0.2)
        y = rng.normal(0, np.exp(h / 2))

        pf = ParticleFilter(
            n_particles=500,
            transition_fn=stochastic_volatility_transition(phi=0.95, sigma_v=0.2),
            observation_loglik=stochastic_volatility_loglik(),
        )
        result = pf.filter(y)
        assert result.filtered_mean.shape == (50,)
        assert np.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Convenience function (DataFrame)
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    def test_basic(self):
        df = _make_df()
        result = particle_filter(
            df,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            n_particles=200,
        )
        assert "filtered_mean" in result.columns
        assert "filtered_var" in result.columns
        assert "ess" in result.columns
        assert len(result) == 50

    def test_multi_group(self):
        df = _make_multi_df()
        result = particle_filter(
            df,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            n_particles=200,
        )
        assert len(result) == 60
        assert sorted(result["unique_id"].unique().to_list()) == ["A", "B"]

    def test_custom_columns(self):
        n = 20
        _, obs = _make_local_level_data(n)
        df = pl.DataFrame(
            {
                "sid": ["X"] * n,
                "ts": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "val": obs.tolist(),
            }
        )
        result = particle_filter(
            df,
            transition_fn=local_level_transition(0.5),
            observation_loglik=local_level_loglik(1.0),
            n_particles=100,
            id_col="sid",
            target_col="val",
            time_col="ts",
        )
        assert "sid" in result.columns
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestResult:
    def test_fields(self):
        r = ParticleFilterResult(
            filtered_mean=np.array([1.0, 2.0]),
            filtered_var=np.array([0.1, 0.2]),
        )
        assert r.particles_history is None
        assert r.log_likelihood == 0.0


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def test_submodule_imports():
    from polars_ts.bayesian.particle_filter import ParticleFilter as PF
    from polars_ts.bayesian.particle_filter import particle_filter as pf

    assert PF is ParticleFilter
    assert pf is particle_filter
