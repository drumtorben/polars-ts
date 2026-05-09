"""Inference methods for Bayesian VAR — analytical posterior and Gibbs sampler."""

from __future__ import annotations

import numpy as np


def _analytical_posterior(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    V0_inv_diag: np.ndarray,
    S0: np.ndarray,
    nu0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute analytical Normal-Wishart posterior.

    Returns
    -------
    B_post
        Posterior mean for coefficients, shape ``(k, k*p+1)``.
    V_post_inv
        Posterior precision, shape ``(k*p+1, k*p+1)``.
    S_post
        Posterior scale matrix, shape ``(k, k)``.
    nu_post
        Posterior degrees of freedom.

    """
    T = X.shape[0]

    # Prior precision as diagonal matrix
    V0_inv = np.diag(V0_inv_diag)

    # Posterior precision
    V_post_inv = V0_inv + X.T @ X

    # Posterior mean
    V_post = np.linalg.inv(V_post_inv)
    B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]  # (k*p+1, k)
    B_post = (V_post @ (V0_inv @ B0.T + X.T @ Y)).T  # (k, k*p+1)

    # Posterior scale
    nu_post = nu0 + T
    resid = Y - X @ B_post.T
    S_post = (
        S0
        + resid.T @ resid
        + (B_ols - B0.T).T @ np.linalg.inv(np.linalg.inv(V0_inv) + np.linalg.inv(X.T @ X)) @ (B_ols - B0.T)
    )

    # Ensure symmetric
    S_post = (S_post + S_post.T) / 2

    return B_post, V_post_inv, S_post, nu_post


def _gibbs_sample(
    X: np.ndarray,
    Y: np.ndarray,
    B0: np.ndarray,
    V0_inv_diag: np.ndarray,
    S0: np.ndarray,
    nu0: float,
    n_samples: int,
    burn_in: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw posterior samples via Gibbs sampling.

    Uses the matrix-normal / inverse-Wishart conjugacy:

    - ``B' | Sigma ~ MN(B_post', V_post, Sigma)`` where
      ``V_post = (V0 + X'X)^{-1}`` and
      ``B_post' = V_post (V0 B0' + X'Y)``
    - Sample via ``B' = B_post' + chol(V_post) Z chol(Sigma)'``
      where ``Z`` is a ``(dim, k)`` standard normal matrix.

    Returns
    -------
    B_samples
        Shape ``(n_samples, k, k*p+1)``.
    Sigma_samples
        Shape ``(n_samples, k, k)``.

    """
    rng = np.random.default_rng(seed)
    T, k = Y.shape
    dim = X.shape[1]

    V0_inv = np.diag(V0_inv_diag)
    XtX = X.T @ X
    XtY = X.T @ Y

    # Posterior for V (shared across Gibbs iterations, doesn't depend on Sigma)
    V_post_inv = V0_inv + XtX
    V_post_inv = (V_post_inv + V_post_inv.T) / 2
    V_post = np.linalg.inv(V_post_inv)
    V_post = (V_post + V_post.T) / 2

    # Posterior mean for B': (dim, k)
    B_post_T = V_post @ (V0_inv @ B0.T + XtY)  # (dim, k)

    # Cholesky of V_post for sampling
    try:
        L_V = np.linalg.cholesky(V_post + np.eye(dim) * 1e-10)
    except np.linalg.LinAlgError:
        L_V = np.diag(np.sqrt(np.maximum(np.diag(V_post), 1e-10)))

    # Initialize Sigma from OLS residuals
    B_ols_T = np.linalg.lstsq(X, Y, rcond=None)[0]  # (dim, k)
    resid = Y - X @ B_ols_T
    Sigma = (resid.T @ resid) / max(T - dim, 1)
    Sigma = (Sigma + Sigma.T) / 2 + np.eye(k) * 1e-8

    total = n_samples + burn_in
    B_samples = np.empty((total, k, dim))
    Sigma_samples = np.empty((total, k, k))

    for i in range(total):
        # --- Sample B | Sigma, Y ---
        try:
            L_Sigma = np.linalg.cholesky(Sigma + np.eye(k) * 1e-10)
        except np.linalg.LinAlgError:
            L_Sigma = np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-10)))

        Z = rng.standard_normal((dim, k))
        B_draw_T = B_post_T + L_V @ Z @ L_Sigma.T  # (dim, k)
        B_draw = B_draw_T.T  # (k, dim)
        B_samples[i] = B_draw

        # --- Sample Sigma | B, Y ---
        resid = Y - X @ B_draw.T
        S_post = S0 + resid.T @ resid
        S_post = (S_post + S_post.T) / 2

        # Draw from Inverse-Wishart(nu_post, S_post)
        nu_post = nu0 + T
        try:
            S_post_inv = np.linalg.inv(S_post)
            S_post_inv = (S_post_inv + S_post_inv.T) / 2
            eigvals = np.linalg.eigvalsh(S_post_inv)
            if eigvals.min() <= 0:
                S_post_inv += np.eye(k) * (abs(eigvals.min()) + 1e-8)
            from scipy.stats import wishart

            Sigma_inv_draw = wishart.rvs(
                df=nu_post,
                scale=S_post_inv / nu_post,
                random_state=rng,
            )
            if k == 1:
                Sigma_inv_draw = np.atleast_2d(Sigma_inv_draw)
            Sigma = np.linalg.inv(Sigma_inv_draw)
            Sigma = (Sigma + Sigma.T) / 2
            eigvals = np.linalg.eigvalsh(Sigma)
            if eigvals.min() <= 0:
                Sigma += np.eye(k) * (abs(eigvals.min()) + 1e-8)
        except (np.linalg.LinAlgError, ValueError):
            pass  # keep previous Sigma

        Sigma_samples[i] = Sigma

    return B_samples[burn_in:], Sigma_samples[burn_in:]
