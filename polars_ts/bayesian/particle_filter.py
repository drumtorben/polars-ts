"""Particle Filter (Sequential Monte Carlo) for nonlinear state estimation.

Implements bootstrap and auxiliary particle filters with systematic
resampling for state estimation in nonlinear, non-Gaussian state-space
models.

References
----------
- Doucet et al. (2001), *Sequential Monte Carlo Methods in Practice*
- Andrieu et al. (2010), *Particle Markov chain Monte Carlo methods*
- Chopin & Papaspiliopoulos (2020), *An Introduction to Sequential Monte Carlo*

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# transition_fn(particles, t, rng) -> new_particles
TransitionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]
# observation_loglik(particles, y_t) -> log_weights
ObservationLogLik = Callable[[np.ndarray, float], np.ndarray]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ParticleFilterResult:
    """Particle filter output.

    Attributes
    ----------
    filtered_mean
        Filtered state mean at each time step, shape ``(T,)`` or ``(T, state_dim)``.
    filtered_var
        Filtered state variance at each time step.
    particles_history
        Full particle trajectories, shape ``(T, n_particles, state_dim)``.
        ``None`` if ``store_history=False``.
    weights_history
        Normalized weights at each step, shape ``(T, n_particles)``.
        ``None`` if ``store_history=False``.
    ess
        Effective sample size at each step, shape ``(T,)``.
    log_likelihood
        Estimate of the log marginal likelihood.

    """

    filtered_mean: np.ndarray
    filtered_var: np.ndarray
    particles_history: np.ndarray | None = None
    weights_history: np.ndarray | None = None
    ess: np.ndarray = field(default_factory=lambda: np.empty(0))
    log_likelihood: float = 0.0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling. Returns index array."""
    n = len(weights)
    positions = (rng.uniform() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, n - 1)


def _multinomial_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling. Returns index array."""
    n = len(weights)
    return rng.choice(n, size=n, p=weights)


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------


def local_level_transition(sigma_level: float = 1.0) -> TransitionFn:
    """Create transition function for local level model."""

    def fn(particles: np.ndarray, _t: int, rng: np.random.Generator) -> np.ndarray:
        return particles + rng.normal(0, sigma_level, size=particles.shape)

    return fn


def local_level_loglik(sigma_obs: float = 1.0) -> ObservationLogLik:
    """Create observation log-likelihood for local level model."""
    log_norm = -0.5 * np.log(2 * np.pi * sigma_obs**2)
    inv_var = 1.0 / (sigma_obs**2)

    def fn(particles: np.ndarray, y: float) -> np.ndarray:
        x = particles.ravel() if particles.ndim > 1 else particles
        return log_norm - 0.5 * (y - x) ** 2 * inv_var

    return fn


def ar1_transition(phi: float = 0.9, sigma: float = 1.0, mu: float = 0.0) -> TransitionFn:
    """Create transition function for AR(1) model."""

    def fn(particles: np.ndarray, _t: int, rng: np.random.Generator) -> np.ndarray:
        return mu + phi * (particles - mu) + rng.normal(0, sigma, size=particles.shape)

    return fn


def stochastic_volatility_transition(phi: float = 0.95, sigma_v: float = 0.2, mu: float = 0.0) -> TransitionFn:
    """Create transition for stochastic volatility model: h_t = mu + phi*(h_{t-1}-mu) + sigma_v*e_t."""

    def fn(particles: np.ndarray, _t: int, rng: np.random.Generator) -> np.ndarray:
        return mu + phi * (particles - mu) + rng.normal(0, sigma_v, size=particles.shape)

    return fn


def stochastic_volatility_loglik() -> ObservationLogLik:
    """Observation log-likelihood for SV model: y_t ~ N(0, exp(h_t))."""

    def fn(particles: np.ndarray, y: float) -> np.ndarray:
        h = particles.ravel() if particles.ndim > 1 else particles
        return -0.5 * h - 0.5 * y**2 * np.exp(-h) - 0.5 * np.log(2 * np.pi)

    return fn


# ---------------------------------------------------------------------------
# Particle Filter class
# ---------------------------------------------------------------------------


class ParticleFilter:
    """Bootstrap Particle Filter for nonlinear state-space models.

    Parameters
    ----------
    n_particles
        Number of particles.
    transition_fn
        State transition function: ``(particles, t, rng) -> new_particles``.
    observation_loglik
        Observation log-likelihood: ``(particles, y_t) -> log_weights``.
    resample_method
        Resampling strategy: ``"systematic"`` or ``"multinomial"``.
    resample_threshold
        Resample when ESS drops below ``resample_threshold * n_particles``.
    seed
        Random seed.
    store_history
        Whether to store full particle/weight histories.

    """

    def __init__(
        self,
        n_particles: int = 1000,
        transition_fn: TransitionFn | None = None,
        observation_loglik: ObservationLogLik | None = None,
        resample_method: str = "systematic",
        resample_threshold: float = 0.5,
        seed: int = 42,
        store_history: bool = False,
    ) -> None:
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        if resample_method not in ("systematic", "multinomial"):
            raise ValueError(f"resample_method must be 'systematic' or 'multinomial', got {resample_method!r}")
        if not 0 < resample_threshold <= 1:
            raise ValueError("resample_threshold must be in (0, 1]")

        self.n_particles = n_particles
        self.transition_fn = transition_fn
        self.observation_loglik = observation_loglik
        self.resample_method = resample_method
        self.resample_threshold = resample_threshold
        self.seed = seed
        self.store_history = store_history

    def filter(
        self,
        observations: np.ndarray,
        initial_particles: np.ndarray | None = None,
    ) -> ParticleFilterResult:
        """Run the particle filter on a sequence of observations.

        Parameters
        ----------
        observations
            Array of observations, shape ``(T,)``.
        initial_particles
            Starting particles, shape ``(n_particles,)``.
            If ``None``, initialized from N(y[0], 1).

        Returns
        -------
        ParticleFilterResult
            Filtering result with means, variances, ESS, and optionally
            full particle/weight histories.

        """
        if self.transition_fn is None:
            raise ValueError("transition_fn must be set before calling filter()")
        if self.observation_loglik is None:
            raise ValueError("observation_loglik must be set before calling filter()")

        rng = np.random.default_rng(self.seed)
        T = len(observations)
        N = self.n_particles

        # Initialize particles
        if initial_particles is not None:
            particles = initial_particles.copy()
        else:
            particles = observations[0] + rng.normal(0, 1, size=N)

        weights = np.ones(N) / N
        resample_fn = _systematic_resample if self.resample_method == "systematic" else _multinomial_resample

        filtered_mean = np.empty(T)
        filtered_var = np.empty(T)
        ess_arr = np.empty(T)
        log_lik = 0.0

        particles_hist = np.empty((T, N)) if self.store_history else None
        weights_hist = np.empty((T, N)) if self.store_history else None

        for t in range(T):
            # Propagate
            if t > 0:
                particles = self.transition_fn(particles, t, rng)

            # Weight
            log_w = self.observation_loglik(particles, observations[t])
            max_log_w = np.max(log_w)
            w = np.exp(log_w - max_log_w)
            w_sum = w.sum()

            if w_sum > 0:
                weights = w / w_sum
                log_lik += max_log_w + np.log(w_sum) - np.log(N)
            else:
                weights = np.ones(N) / N

            # Compute statistics
            filtered_mean[t] = np.average(particles, weights=weights)
            filtered_var[t] = np.average((particles - filtered_mean[t]) ** 2, weights=weights)
            ess = 1.0 / np.sum(weights**2)
            ess_arr[t] = ess

            # Store history
            if self.store_history and particles_hist is not None and weights_hist is not None:
                particles_hist[t] = particles.copy()
                weights_hist[t] = weights.copy()

            # Resample if ESS too low
            if ess < self.resample_threshold * N:
                indices = resample_fn(weights, rng)
                particles = particles[indices]
                weights = np.ones(N) / N

        return ParticleFilterResult(
            filtered_mean=filtered_mean,
            filtered_var=filtered_var,
            particles_history=particles_hist,
            weights_history=weights_hist,
            ess=ess_arr,
            log_likelihood=log_lik,
        )


# ---------------------------------------------------------------------------
# Convenience function for DataFrame input
# ---------------------------------------------------------------------------


def particle_filter(
    df: pl.DataFrame,
    transition_fn: TransitionFn,
    observation_loglik: ObservationLogLik,
    n_particles: int = 1000,
    resample_method: str = "systematic",
    resample_threshold: float = 0.5,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Particle filter convenience function for DataFrames.

    Parameters
    ----------
    df
        Input DataFrame.
    transition_fn
        State transition function.
    observation_loglik
        Observation log-likelihood function.
    n_particles
        Number of particles.
    resample_method
        ``"systematic"`` or ``"multinomial"``.
    resample_threshold
        ESS threshold for resampling.
    seed
        Random seed.
    id_col, target_col, time_col
        Column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``filtered_mean``, ``filtered_var``, ``ess`` columns.

    """
    pf = ParticleFilter(
        n_particles=n_particles,
        transition_fn=transition_fn,
        observation_loglik=observation_loglik,
        resample_method=resample_method,
        resample_threshold=resample_threshold,
        seed=seed,
    )

    sorted_df = df.sort(id_col, time_col)
    all_rows: list[dict[str, Any]] = []

    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        y = group_df[target_col].to_numpy().astype(np.float64)
        result = pf.filter(y)

        for t in range(len(y)):
            all_rows.append(
                {
                    id_col: gid,
                    "t": t,
                    "filtered_mean": float(result.filtered_mean[t]),
                    "filtered_var": float(result.filtered_var[t]),
                    "ess": float(result.ess[t]),
                }
            )

    return pl.DataFrame(all_rows)
