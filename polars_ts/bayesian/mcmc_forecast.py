"""Posterior predictive forecast functions for MCMC models."""

from __future__ import annotations

import numpy as np


def _forecast_local_level(y: np.ndarray, samples: np.ndarray, h: int, seed: int) -> np.ndarray:
    """Posterior predictive forecast for local level model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma_obs = abs(samples[i, 0])
        sigma_level = abs(samples[i, 1])
        level = samples[i, 2]

        for t in range(len(y)):
            alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2 + 1e-20)
            level = alpha * y[t] + (1 - alpha) * level

        for step in range(h):
            level += rng.normal(0, sigma_level)
            forecasts[i, step] = level + rng.normal(0, sigma_obs)

    return forecasts


def _forecast_ar(y: np.ndarray, samples: np.ndarray, h: int, p: int, seed: int) -> np.ndarray:
    """Posterior predictive forecast for AR(p) model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma = abs(samples[i, 0])
        mu = samples[i, 1]
        phi = samples[i, 2 : 2 + p]

        history = list(y[-p:])
        for step in range(h):
            pred = mu
            for j in range(p):
                pred += phi[j] * (history[-(j + 1)] - mu)
            pred += rng.normal(0, sigma)
            forecasts[i, step] = pred
            history.append(pred)

    return forecasts


def _forecast_seasonal(y: np.ndarray, samples: np.ndarray, h: int, season_length: int, seed: int) -> np.ndarray:
    """Posterior predictive forecast for seasonal local level model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    n = len(y)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma_obs = abs(samples[i, 0])
        sigma_level = abs(samples[i, 1])
        sigma_season = abs(samples[i, 2])
        level = samples[i, 3]
        seasons = list(samples[i, 4 : 4 + season_length])

        for t in range(n):
            s_idx = t % season_length
            alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2 + 1e-20)
            level = alpha * (y[t] - seasons[s_idx]) + (1 - alpha) * level
            gamma = sigma_season**2 / (sigma_season**2 + sigma_obs**2 + 1e-20)
            seasons[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasons[s_idx]

        for step in range(h):
            s_idx = (n + step) % season_length
            level += rng.normal(0, sigma_level)
            seasons[s_idx] += rng.normal(0, sigma_season)
            forecasts[i, step] = level + seasons[s_idx] + rng.normal(0, sigma_obs)

    return forecasts
