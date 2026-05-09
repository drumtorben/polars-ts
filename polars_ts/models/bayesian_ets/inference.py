"""Likelihood, parameter packing, log-posterior, MAP, MCMC, and forecasting for Bayesian ETS."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from polars_ts.models.bayesian_ets.priors import (
    ETSPriors,
    ModelType,
    _log_prior_invgamma,
    _log_prior_normal,
    _log_prior_smoothing,
)


def _ses_loglik(values: list[float], alpha: float, level0: float, sigma: float) -> float:
    """Gaussian log-likelihood for SES state-space model."""
    if sigma <= 0:
        return -np.inf
    n = len(values)
    level = level0
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma**2)
    inv_sigma = 1.0 / sigma
    for t in range(n):
        residual = values[t] - level
        ll += log_norm - 0.5 * (residual * inv_sigma) ** 2
        level = alpha * values[t] + (1 - alpha) * level
    return ll


def _holt_loglik(values: list[float], alpha: float, beta: float, level0: float, trend0: float, sigma: float) -> float:
    """Gaussian log-likelihood for Holt's linear trend model."""
    if sigma <= 0:
        return -np.inf
    n = len(values)
    level = level0
    trend = trend0
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma**2)
    inv_sigma = 1.0 / sigma
    for t in range(n):
        predicted = level + trend
        residual = values[t] - predicted
        ll += log_norm - 0.5 * (residual * inv_sigma) ** 2
        prev_level = level
        level = alpha * values[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return ll


def _hw_loglik(
    values: list[float],
    alpha: float,
    beta: float,
    gamma: float,
    level0: float,
    trend0: float,
    seasons0: list[float],
    m: int,
    additive: bool,
    sigma: float,
) -> float:
    """Gaussian log-likelihood for Holt-Winters model."""
    if sigma <= 0:
        return -np.inf
    n = len(values)
    level = level0
    trend = trend0
    seasons = list(seasons0)
    ll = 0.0
    log_norm = -0.5 * math.log(2 * math.pi * sigma**2)
    inv_sigma = 1.0 / sigma

    for t in range(n):
        s_idx = t % m
        if additive:
            predicted = level + trend + seasons[s_idx]
        else:
            predicted = (level + trend) * seasons[s_idx]

        residual = values[t] - predicted
        ll += log_norm - 0.5 * (residual * inv_sigma) ** 2

        prev_level = level
        if additive:
            level = alpha * (values[t] - seasons[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasons[s_idx] = gamma * (values[t] - level) + (1 - gamma) * seasons[s_idx]
        else:
            denom_s = seasons[s_idx] if seasons[s_idx] != 0 else 1.0
            denom_l = level if level != 0 else 1.0
            level = alpha * (values[t] / denom_s) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasons[s_idx] = gamma * (values[t] / denom_l) + (1 - gamma) * seasons[s_idx]

    return ll


def _pack_params(
    model: ModelType,
    alpha: float,
    beta: float | None,
    gamma: float | None,
    level0: float,
    trend0: float | None,
    seasons0: list[float] | None,
    sigma: float,
) -> np.ndarray:
    """Pack parameters into a flat array for optimization."""
    params: list[float] = [alpha]
    if model in ("holt", "holt_winters"):
        params.append(beta if beta is not None else 0.1)
    if model == "holt_winters":
        params.append(gamma if gamma is not None else 0.1)
    params.append(level0)
    if model in ("holt", "holt_winters"):
        params.append(trend0 if trend0 is not None else 0.0)
    if model == "holt_winters" and seasons0 is not None:
        params.extend(seasons0)
    params.append(sigma)
    return np.array(params)


def _unpack_params(theta: np.ndarray, model: ModelType, m: int) -> dict[str, Any]:
    """Unpack flat parameter array into named parameters."""
    idx = 0
    alpha = theta[idx]
    idx += 1

    beta = None
    if model in ("holt", "holt_winters"):
        beta = theta[idx]
        idx += 1

    gamma = None
    if model == "holt_winters":
        gamma = theta[idx]
        idx += 1

    level0 = theta[idx]
    idx += 1

    trend0 = None
    if model in ("holt", "holt_winters"):
        trend0 = theta[idx]
        idx += 1

    seasons0 = None
    if model == "holt_winters":
        seasons0 = theta[idx : idx + m].tolist()
        idx += m

    sigma = theta[idx]
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "level0": level0,
        "trend0": trend0,
        "seasons0": seasons0,
        "sigma": sigma,
    }


def _log_posterior(
    theta: np.ndarray,
    values: list[float],
    model: ModelType,
    m: int,
    additive: bool,
    priors: ETSPriors,
) -> float:
    """Compute unnormalized log-posterior."""
    p = _unpack_params(theta, model, m)

    for key in ("alpha", "beta", "gamma"):
        val = p[key]
        if val is not None and (val <= 0 or val >= 1):
            return -np.inf
    if p["sigma"] <= 0:
        return -np.inf

    if model == "ses":
        ll = _ses_loglik(values, p["alpha"], p["level0"], p["sigma"])
    elif model == "holt":
        ll = _holt_loglik(values, p["alpha"], p["beta"], p["level0"], p["trend0"], p["sigma"])
    else:
        ll = _hw_loglik(
            values,
            p["alpha"],
            p["beta"],
            p["gamma"],
            p["level0"],
            p["trend0"],
            p["seasons0"],
            m,
            additive,
            p["sigma"],
        )

    if not np.isfinite(ll):
        return -np.inf

    lp = 0.0
    lp += _log_prior_smoothing(p["alpha"], priors.alpha_a, priors.alpha_b)
    if p["beta"] is not None:
        lp += _log_prior_smoothing(p["beta"], priors.beta_a, priors.beta_b)
    if p["gamma"] is not None:
        lp += _log_prior_smoothing(p["gamma"], priors.gamma_a, priors.gamma_b)
    lp += _log_prior_normal(p["level0"], priors.level_mu, priors.level_sigma)
    if p["trend0"] is not None:
        lp += _log_prior_normal(p["trend0"], priors.trend_mu, priors.trend_sigma)
    lp += _log_prior_invgamma(p["sigma"], priors.sigma_shape, priors.sigma_scale)

    return ll + lp


def _map_estimate(values: list[float], model: ModelType, m: int, additive: bool, priors: ETSPriors) -> np.ndarray:
    """Find MAP estimate via L-BFGS-B."""
    alpha0 = 0.3
    beta0 = 0.1
    gamma0 = 0.1
    level0_init = float(np.mean(values))
    trend0_init = float((values[-1] - values[0]) / max(len(values) - 1, 1)) if len(values) > 1 else 0.0
    std_val = float(np.std(values))
    sigma0 = std_val if std_val > 0 else 1.0

    seasons0_init: list[float] | None = None
    if model == "holt_winters" and len(values) >= m:
        first_season_avg = float(np.mean(values[:m]))
        if additive:
            seasons0_init = [values[i] - first_season_avg for i in range(m)]
        else:
            seasons0_init = [values[i] / first_season_avg if first_season_avg != 0 else 1.0 for i in range(m)]
    elif model == "holt_winters":
        seasons0_init = [0.0] * m

    x0 = _pack_params(model, alpha0, beta0, gamma0, level0_init, trend0_init, seasons0_init, sigma0)

    eps = 1e-6
    bounds: list[tuple[float | None, float | None]] = [(eps, 1 - eps)]  # alpha
    if model in ("holt", "holt_winters"):
        bounds.append((eps, 1 - eps))
    if model == "holt_winters":
        bounds.append((eps, 1 - eps))
    bounds.append((None, None))  # level0
    if model in ("holt", "holt_winters"):
        bounds.append((None, None))
    if model == "holt_winters":
        bounds.extend([(None, None)] * m)
    bounds.append((eps, None))  # sigma

    def neg_log_post(theta: np.ndarray) -> float:
        val = _log_posterior(theta, values, model, m, additive, priors)
        return -val if np.isfinite(val) else 1e20

    from scipy.optimize import minimize

    result = minimize(neg_log_post, x0, method="L-BFGS-B", bounds=bounds)
    return result.x


def _mcmc_sample(
    values: list[float],
    model: ModelType,
    m: int,
    additive: bool,
    priors: ETSPriors,
    n_samples: int = 1000,
    burn_in: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Draw posterior samples via Metropolis-Hastings."""
    rng = np.random.default_rng(seed)
    theta_current = _map_estimate(values, model, m, additive, priors)
    n_params = len(theta_current)
    lp_current = _log_posterior(theta_current, values, model, m, additive, priors)

    proposal_scale = np.abs(theta_current) * 0.01
    proposal_scale = np.maximum(proposal_scale, 1e-4)

    total = n_samples + burn_in
    samples = np.empty((total, n_params))

    for i in range(total):
        theta_proposal = theta_current + rng.normal(0, proposal_scale)
        lp_proposal = _log_posterior(theta_proposal, values, model, m, additive, priors)
        log_ratio = lp_proposal - lp_current
        if np.isfinite(log_ratio) and math.log(rng.uniform()) < log_ratio:
            theta_current = theta_proposal
            lp_current = lp_proposal
        samples[i] = theta_current

    return samples[burn_in:]


def _forecast_from_params(
    values: list[float],
    params: dict[str, Any],
    model: ModelType,
    m: int,
    additive: bool,
    h: int,
    sigma_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Run ETS forward to get h-step forecasts from fitted parameters."""
    alpha = params["alpha"]

    if model == "ses":
        level = params["level0"]
        for v in values:
            level = alpha * v + (1 - alpha) * level
        forecasts = [level] * h

    elif model == "holt":
        beta = params["beta"]
        level = params["level0"]
        trend = params["trend0"]
        for v in values:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        forecasts = [level + step * trend for step in range(1, h + 1)]

    else:  # holt_winters
        beta = params["beta"]
        gamma = params["gamma"]
        level = params["level0"]
        trend = params["trend0"]
        seasons = list(params["seasons0"])
        n = len(values)

        for t in range(n):
            s_idx = t % m
            prev_level = level
            if additive:
                level = alpha * (values[t] - seasons[s_idx]) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasons[s_idx] = gamma * (values[t] - level) + (1 - gamma) * seasons[s_idx]
            else:
                denom_s = seasons[s_idx] if seasons[s_idx] != 0 else 1.0
                denom_l = level if level != 0 else 1.0
                level = alpha * (values[t] / denom_s) + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                seasons[s_idx] = gamma * (values[t] / denom_l) + (1 - gamma) * seasons[s_idx]

        forecasts = []
        for step in range(1, h + 1):
            s_idx = (n - 1 + step) % m
            if additive:
                forecasts.append(level + step * trend + seasons[s_idx])
            else:
                forecasts.append((level + step * trend) * seasons[s_idx])

    if sigma_noise and rng is not None:
        sigma = params["sigma"]
        forecasts = [f + rng.normal(0, sigma) for f in forecasts]

    return forecasts
