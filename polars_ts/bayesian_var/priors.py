"""Prior specifications for Bayesian VAR.

Provides the Minnesota (Litterman) and Normal-Wishart prior dataclasses,
along with helper functions for building prior precision matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MinnesotaPrior:
    """Minnesota (Litterman) prior for BVAR.

    Shrinks VAR coefficients toward a random walk: own first lag
    receives prior mean 1, all others 0. Tightness parameters
    control how strongly the prior pulls toward this structure.

    Parameters
    ----------
    lambda1
        Overall tightness. Smaller values shrink more aggressively.
    lambda2
        Cross-variable tightness (relative to own-lag). Typically < 1
        so cross-variable lags are shrunk harder.
    lambda3
        Lag decay. Higher values shrink distant lags more aggressively.
        Prior variance for lag *l* is scaled by ``l^{-lambda3}``.
    sigma_scale
        Per-variable residual variance estimates. If ``None``, estimated
        from univariate AR(p) regressions.

    """

    lambda1: float = 0.2
    lambda2: float = 0.5
    lambda3: float = 1.0
    sigma_scale: np.ndarray | None = None


@dataclass
class NormalWishartPrior:
    """Normal-Wishart conjugate prior for BVAR.

    Places a matrix-normal prior on the coefficient matrix ``B``
    and a Wishart prior on the precision matrix ``Sigma^{-1}``.

    Parameters
    ----------
    B0
        Prior mean for the coefficient matrix, shape ``(k, k*p+1)``.
        If ``None``, defaults to random walk (identity on first own-lag).
    V0
        Prior precision (inverse covariance) for vec(B),
        shape ``(k*p+1, k*p+1)``. If ``None``, uses Minnesota-style
        diagonal with the given ``tightness``.
    S0
        Prior scale matrix for Wishart, shape ``(k, k)``.
        If ``None``, uses identity scaled by data variance.
    nu0
        Degrees of freedom for Wishart. Must be >= k.
        If ``None``, defaults to ``k + 2``.
    tightness
        Diagonal tightness for automatic V0 construction.

    """

    B0: np.ndarray | None = None
    V0: np.ndarray | None = None
    S0: np.ndarray | None = None
    nu0: float | None = None
    tightness: float = 0.1


def _estimate_sigma_from_ar(data: np.ndarray, p: int) -> np.ndarray:
    """Estimate per-variable residual variance from univariate AR(p)."""
    _n, k = data.shape
    sigmas = np.ones(k)
    for j in range(k):
        y_j = data[:, j]
        n_j = len(y_j)
        if n_j <= p + 1:
            continue
        X_ar = np.column_stack([y_j[p - i - 1 : n_j - i - 1] for i in range(p)] + [np.ones(n_j - p)])
        Y_ar = y_j[p:]
        beta = np.linalg.lstsq(X_ar, Y_ar, rcond=None)[0]
        resid = Y_ar - X_ar @ beta
        sigmas[j] = max(float(np.var(resid, ddof=p + 1)), 1e-10)
    return sigmas


def _minnesota_prior_precision(
    k: int,
    p: int,
    prior: MinnesotaPrior,
    sigma_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Minnesota prior mean B0 and precision V0_inv.

    Returns
    -------
    B0
        Prior mean, shape ``(k, k*p+1)``.
    V0_inv
        Prior precision diagonal, shape ``(k*p+1,)``.

    """
    dim = k * p + 1
    B0 = np.zeros((k, dim))
    # Random walk: own first lag = 1
    for j in range(k):
        B0[j, j] = 1.0

    V0_inv_diag = np.zeros(dim)
    for lag in range(1, p + 1):
        for j in range(k):
            col_idx = (lag - 1) * k + j
            # Own lag
            var_own = (prior.lambda1 / (lag**prior.lambda3)) ** 2
            V0_inv_diag[col_idx] = 1.0 / max(var_own, 1e-20)
            # Cross lags get tighter shrinkage
            for i in range(k):
                if i != j:
                    # Rescale by relative variance
                    s_ratio = sigma_scale[i] / max(sigma_scale[j], 1e-20)
                    var_cross = (prior.lambda1 * prior.lambda2 / (lag**prior.lambda3)) ** 2 * s_ratio
                    # This affects the prior for equation i, lag of variable j
                    # We set the diagonal for the coefficient in equation i
                    # but here we build a shared V0, so use average
                    V0_inv_diag[col_idx] = max(V0_inv_diag[col_idx], 1.0 / max(var_cross, 1e-20))

    # Intercept: diffuse
    V0_inv_diag[-1] = 1e-6

    return B0, V0_inv_diag
