"""Built-in log-posteriors and Metropolis-Hastings sampler for MCMC forecasting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _local_level_logpost(params: np.ndarray, y: np.ndarray) -> float:
    """Log-posterior for local level model: y_t = level_t + eps, level_t = level_{t-1} + eta."""
    sigma_obs = params[0]
    sigma_level = params[1]
    level0 = params[2]

    if sigma_obs <= 0 or sigma_level <= 0:
        return -np.inf

    n = len(y)
    level = level0
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma_obs**2)
    inv_s = 1.0 / sigma_obs

    for t in range(n):
        ll += log_norm - 0.5 * ((y[t] - level) * inv_s) ** 2
        level = level + sigma_level * 0  # deterministic forward for loglik
        alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2)
        level = alpha * y[t] + (1 - alpha) * level

    lp = -0.5 * (level0 / 100.0) ** 2
    lp += -0.5 * (sigma_obs / 10.0) ** 2
    lp += -0.5 * (sigma_level / 10.0) ** 2

    return ll + lp


def _ar_logpost(params: np.ndarray, y: np.ndarray, p: int) -> float:
    """Log-posterior for AR(p) model."""
    sigma = params[0]
    mu = params[1]
    phi = params[2 : 2 + p]

    if sigma <= 0:
        return -np.inf

    n = len(y)
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma**2)
    inv_s = 1.0 / sigma

    for t in range(p, n):
        pred = mu
        for j in range(p):
            pred += phi[j] * (y[t - j - 1] - mu)
        ll += log_norm - 0.5 * ((y[t] - pred) * inv_s) ** 2

    lp = -0.5 * (sigma / 10.0) ** 2
    lp += -0.5 * (mu / 100.0) ** 2
    for j in range(p):
        lp += -0.5 * phi[j] ** 2

    return ll + lp


def _seasonal_logpost(params: np.ndarray, y: np.ndarray, season_length: int) -> float:
    """Log-posterior for seasonal local level model."""
    sigma_obs = params[0]
    sigma_level = params[1]
    sigma_season = params[2]
    level0 = params[3]
    seasons = params[4 : 4 + season_length]

    if sigma_obs <= 0 or sigma_level <= 0 or sigma_season <= 0:
        return -np.inf

    n = len(y)
    level = level0
    s = list(seasons)
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma_obs**2)
    inv_s = 1.0 / sigma_obs

    for t in range(n):
        s_idx = t % season_length
        pred = level + s[s_idx]
        ll += log_norm - 0.5 * ((y[t] - pred) * inv_s) ** 2
        alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2)
        level = alpha * (y[t] - s[s_idx]) + (1 - alpha) * level
        gamma = sigma_season**2 / (sigma_season**2 + sigma_obs**2)
        s[s_idx] = gamma * (y[t] - level) + (1 - gamma) * s[s_idx]

    lp = -0.5 * (level0 / 100.0) ** 2
    lp += -0.5 * (sigma_obs / 10.0) ** 2
    lp += -0.5 * (sigma_level / 10.0) ** 2
    lp += -0.5 * (sigma_season / 10.0) ** 2
    for si in seasons:
        lp += -0.5 * (si / 10.0) ** 2

    return ll + lp


def _mh_sample(
    logpost_fn: Any,
    x0: np.ndarray,
    n_samples: int,
    burn_in: int,
    seed: int,
) -> np.ndarray:
    """Metropolis-Hastings sampler. Returns (n_samples, n_params)."""
    rng = np.random.default_rng(seed)
    n_params = len(x0)

    theta = x0.copy()
    lp = logpost_fn(theta)

    proposal_scale = np.abs(theta) * 0.02
    proposal_scale = np.maximum(proposal_scale, 1e-4)

    total = n_samples + burn_in
    samples = np.empty((total, n_params))

    for i in range(total):
        proposal = theta + rng.normal(0, proposal_scale)
        lp_prop = logpost_fn(proposal)

        log_ratio = lp_prop - lp
        if np.isfinite(log_ratio) and math.log(rng.uniform()) < log_ratio:
            theta = proposal
            lp = lp_prop

        samples[i] = theta

    return samples[burn_in:]
