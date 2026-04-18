"""Bayesian Online Changepoint Detection (BOCPD)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def bocpd(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    hazard_rate: float = 200.0,
    mu_prior: float = 0.0,
    kappa_prior: float = 1.0,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Detect changepoints using Bayesian Online Changepoint Detection.

    Uses a normal-inverse-gamma conjugate model with constant hazard.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to analyze.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    hazard_rate
        Expected run length (higher = fewer changepoints).
    mu_prior
        Prior mean for the normal model.
    kappa_prior
        Prior precision scaling.
    alpha_prior
        Prior shape for inverse-gamma.
    beta_prior
        Prior rate for inverse-gamma.
    threshold
        Probability threshold above which a changepoint is flagged.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, time_col, "run_length", "changepoint_prob"]``
        and a boolean ``"is_changepoint"`` column.

    """
    if hazard_rate <= 0:
        raise ValueError("hazard_rate must be positive")

    sorted_df = df.sort(id_col, time_col)
    h = 1.0 / hazard_rate  # Constant hazard function

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        data = np.array(group_df[target_col].to_list(), dtype=np.float64)
        times = group_df[time_col].to_list()
        n = len(data)

        # Sufficient statistics for each run length
        mu = np.array([mu_prior])
        kappa = np.array([kappa_prior])
        alpha = np.array([alpha_prior])
        beta_arr = np.array([beta_prior])

        # Run-length probabilities
        r_probs = np.array([1.0])

        for t in range(n):
            x = data[t]

            # Predictive probability under each run length (Student-t)
            df_t = 2 * alpha
            scale = beta_arr * (kappa + 1) / (alpha * kappa)
            pred_probs = np.exp(_log_student_t(x, mu, scale, df_t))

            # Growth probabilities
            growth = r_probs * pred_probs * (1 - h)

            # Changepoint probability
            cp_prob = float(np.sum(r_probs * pred_probs * h))

            # New run-length distribution
            new_r = np.empty(len(growth) + 1)
            new_r[0] = cp_prob
            new_r[1:] = growth

            # Normalize
            total = new_r.sum()
            if total > 0:
                new_r /= total

            # Most probable run length
            run_length = int(np.argmax(new_r))

            rows.append(
                {
                    id_col: gid,
                    time_col: times[t],
                    "run_length": run_length,
                    "changepoint_prob": float(new_r[0]),
                    "is_changepoint": float(new_r[0]) > threshold,
                }
            )

            r_probs = new_r

            # Update sufficient statistics
            new_mu = np.empty(len(mu) + 1)
            new_kappa = np.empty(len(kappa) + 1)
            new_alpha = np.empty(len(alpha) + 1)
            new_beta = np.empty(len(beta_arr) + 1)

            new_mu[0] = mu_prior
            new_kappa[0] = kappa_prior
            new_alpha[0] = alpha_prior
            new_beta[0] = beta_prior

            new_mu[1:] = (kappa * mu + x) / (kappa + 1)
            new_kappa[1:] = kappa + 1
            new_alpha[1:] = alpha + 0.5
            new_beta[1:] = beta_arr + kappa * (x - mu) ** 2 / (2 * (kappa + 1))

            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta_arr = new_beta

    return pl.DataFrame(rows)


def _log_student_t(x: float, mu: np.ndarray, scale: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Log-probability of x under a Student-t distribution."""
    from scipy.special import gammaln

    z = (x - mu) ** 2 / scale
    log_p = (
        gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale) - (df + 1) / 2 * np.log(1 + z / df)
    )
    return log_p
