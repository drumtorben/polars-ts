"""BayesianVARResult — fitted result container for Bayesian VAR."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BayesianVARResult:
    """Fitted Bayesian VAR result.

    Attributes
    ----------
    B_post
        Posterior mean coefficient matrix, shape ``(k, k*p+1)``.
    Sigma_post
        Posterior mean covariance matrix, shape ``(k, k)``.
    B_samples
        MCMC posterior samples for B, shape ``(n_samples, k, k*p+1)``.
        ``None`` for analytical inference.
    Sigma_samples
        MCMC posterior samples for Sigma, shape ``(n_samples, k, k)``.
        ``None`` for analytical inference.
    target_cols
        Names of the modeled variables.
    p
        Number of lags.

    """

    B_post: np.ndarray
    Sigma_post: np.ndarray
    B_samples: np.ndarray | None = None
    Sigma_samples: np.ndarray | None = None
    target_cols: list[str] = field(default_factory=list)
    p: int = 1
    _last_values: np.ndarray = field(default_factory=lambda: np.empty(0))
