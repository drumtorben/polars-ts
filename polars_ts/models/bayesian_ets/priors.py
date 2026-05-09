"""Prior specifications and log-prior functions for Bayesian ETS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ModelType = Literal["ses", "holt", "holt_winters"]
InferenceMethod = Literal["map", "mcmc"]


@dataclass
class ETSPriors:
    """Prior distributions for ETS smoothing parameters and initial states."""

    alpha_a: float = 2.0
    alpha_b: float = 2.0
    beta_a: float = 2.0
    beta_b: float = 2.0
    gamma_a: float = 2.0
    gamma_b: float = 2.0
    level_mu: float = 0.0
    level_sigma: float = 100.0
    trend_mu: float = 0.0
    trend_sigma: float = 10.0
    sigma_shape: float = 2.0
    sigma_scale: float = 1.0


def _log_prior_smoothing(value: float, a: float, b: float) -> float:
    """Log-density of Beta(a, b) prior, clamped to valid domain."""
    if value <= 0 or value >= 1:
        return -np.inf
    from scipy.stats import beta

    return beta.logpdf(value, a, b)


def _log_prior_normal(value: float, mu: float, sigma: float) -> float:
    from scipy.stats import norm

    return norm.logpdf(value, mu, sigma)


def _log_prior_invgamma(value: float, shape: float, scale: float) -> float:
    if value <= 0:
        return -np.inf
    from scipy.stats import invgamma

    return invgamma.logpdf(value, shape, scale=scale)
