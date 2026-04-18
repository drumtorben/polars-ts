"""Regime detection via Gaussian Hidden Markov Model."""

from __future__ import annotations

import numpy as np
import polars as pl


def regime_detect(
    df: pl.DataFrame,
    n_states: int = 2,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> pl.DataFrame:
    """Detect latent regimes using a Gaussian HMM (Baum-Welch).

    Parameters
    ----------
    df
        Input DataFrame.
    n_states
        Number of hidden states (regimes).
    target_col
        Column to analyze.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    max_iter
        Maximum EM iterations.
    tol
        Convergence tolerance on log-likelihood.
    seed
        Random seed for initialization.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``"regime"`` column (integer state assignment)
        and ``"regime_prob"`` (probability of the assigned state).

    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    sorted_df = df.sort(id_col, time_col)
    rng = np.random.default_rng(seed)

    result_frames: list[pl.DataFrame] = []
    for _gid, group_df in sorted_df.group_by(id_col, maintain_order=True):
        data = np.array(group_df[target_col].to_list(), dtype=np.float64)
        n = len(data)
        k = n_states

        # Initialize parameters via k-means-like splitting
        sorted_data = np.sort(data)
        split_points = np.linspace(0, n, k + 1, dtype=int)
        means = np.array([sorted_data[split_points[i] : split_points[i + 1]].mean() for i in range(k)])
        variances = np.full(k, float(np.var(data)) + 1e-6)
        pi = np.ones(k) / k  # Initial state distribution
        trans = np.full((k, k), 1.0 / k)  # Transition matrix

        # Add small perturbation
        means += rng.normal(0, 0.01, k)

        log_lik_prev = -np.inf

        for _iteration in range(max_iter):
            # E-step: forward-backward
            # Emission probabilities
            log_emit = np.zeros((n, k))
            for j in range(k):
                log_emit[:, j] = -0.5 * np.log(2 * np.pi * variances[j]) - (data - means[j]) ** 2 / (2 * variances[j])

            # Forward pass (log-scale)
            log_alpha = np.zeros((n, k))
            log_alpha[0] = np.log(pi + 1e-300) + log_emit[0]
            for t in range(1, n):
                for j in range(k):
                    log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + np.log(trans[:, j] + 1e-300)) + log_emit[t, j]

            # Backward pass
            log_beta = np.zeros((n, k))
            for t in range(n - 2, -1, -1):
                for j in range(k):
                    log_beta[t, j] = _logsumexp(np.log(trans[j, :] + 1e-300) + log_emit[t + 1] + log_beta[t + 1])

            # Posterior (gamma)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Log-likelihood
            log_lik = _logsumexp(log_alpha[-1])
            if abs(log_lik - log_lik_prev) < tol:
                break
            log_lik_prev = log_lik

            # M-step
            gamma_sum = gamma.sum(axis=0) + 1e-300
            pi = gamma[0] / gamma[0].sum()

            for j in range(k):
                means[j] = np.dot(gamma[:, j], data) / gamma_sum[j]
                variances[j] = np.dot(gamma[:, j], (data - means[j]) ** 2) / gamma_sum[j] + 1e-6

            # Transition matrix
            for i in range(k):
                for j in range(k):
                    xi_sum = 0.0
                    for t in range(n - 1):
                        xi_sum += np.exp(
                            log_alpha[t, i]
                            + np.log(trans[i, j] + 1e-300)
                            + log_emit[t + 1, j]
                            + log_beta[t + 1, j]
                            - log_lik
                        )
                    trans[i, j] = xi_sum
                row_sum = trans[i].sum()
                if row_sum > 0:
                    trans[i] /= row_sum

        # Viterbi decoding for MAP state sequence
        states = np.argmax(gamma, axis=1)
        probs = np.max(gamma, axis=1)

        result_frames.append(
            group_df.with_columns(
                pl.Series("regime", states.tolist()).cast(pl.Int64),
                pl.Series("regime_prob", probs.tolist()),
            )
        )

    return pl.concat(result_frames)


def _logsumexp(
    a: np.ndarray,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray | float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out  # type: ignore[return-value]
