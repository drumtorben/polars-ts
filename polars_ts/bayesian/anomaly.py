"""Bayesian anomaly scoring via posterior predictive p-values.

Scores each observation using an online conjugate normal model that
maintains a running posterior over mean and variance, then computes
tail-area probabilities and Bayes factors.

References
----------
- Gelman et al. (2013), *Bayesian Data Analysis*, Chapter 6
- Adams & MacKay (2007), *Bayesian Online Changepoint Detection*

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BayesianAnomalyResult:
    """Container for Bayesian anomaly scoring output.

    Attributes
    ----------
    scores
        DataFrame with anomaly scores per observation.
    n_anomalies
        Total number of flagged anomalies.

    """

    scores: pl.DataFrame
    n_anomalies: int = 0


# ---------------------------------------------------------------------------
# Online conjugate normal model (Normal-Inverse-Gamma)
# ---------------------------------------------------------------------------


@dataclass
class _NIGState:
    """Normal-Inverse-Gamma sufficient statistics for online updating.

    Prior: mu | sigma^2 ~ N(mu0, sigma^2 / kappa0)
           sigma^2 ~ IG(alpha0, beta0)
    """

    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0

    def update(self, y: float) -> None:
        """Update posterior with a single observation."""
        kappa_new = self.kappa + 1
        mu_new = (self.kappa * self.mu + y) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.kappa * (y - self.mu) ** 2 / kappa_new
        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def predictive_params(self) -> tuple[float, float]:
        """Return (mean, scale) of the posterior predictive t-distribution."""
        pred_mean = self.mu
        pred_scale = math.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return pred_mean, pred_scale

    def predictive_df(self) -> float:
        """Degrees of freedom for the posterior predictive t-distribution."""
        return 2.0 * self.alpha


def _t_cdf(x: float, df: float) -> float:
    """CDF of Student's t-distribution using scipy (lazy import)."""
    from scipy.stats import t

    return float(t.cdf(x, df))


def _compute_pvalue(y: float, state: _NIGState) -> float:
    """Compute two-sided posterior predictive p-value."""
    pred_mean, pred_scale = state.predictive_params()
    df = state.predictive_df()

    if pred_scale <= 0:
        return 1.0

    t_stat = (y - pred_mean) / pred_scale
    p_left = _t_cdf(t_stat, df)
    return 2.0 * min(p_left, 1.0 - p_left)


def _compute_bayes_factor(y: float, state: _NIGState, anomaly_scale: float = 10.0) -> float:
    """Bayes factor: evidence for normal model vs anomaly model.

    H0: y ~ predictive(state)
    H1: y ~ predictive(state, scale * anomaly_scale)

    Returns BF01 (>1 favors normal, <1 favors anomaly).
    """
    from scipy.stats import t

    pred_mean, pred_scale = state.predictive_params()
    df = state.predictive_df()

    if pred_scale <= 0:
        return 1.0

    log_p0 = t.logpdf((y - pred_mean) / pred_scale, df) - math.log(pred_scale)
    log_p1 = t.logpdf((y - pred_mean) / (pred_scale * anomaly_scale), df) - math.log(pred_scale * anomaly_scale)

    log_bf = log_p0 - log_p1
    return float(np.exp(np.clip(log_bf, -500, 500)))


# ---------------------------------------------------------------------------
# BayesianAnomalyDetector class
# ---------------------------------------------------------------------------


class BayesianAnomalyDetector:
    """Bayesian anomaly detector using posterior predictive p-values.

    Maintains an online Normal-Inverse-Gamma conjugate model per series,
    scoring each observation via its posterior predictive tail probability.

    Parameters
    ----------
    threshold
        P-value threshold below which an observation is flagged (default 0.01).
    prior_mu
        Prior mean for the normal model. If ``None``, uses first observation.
    prior_kappa
        Prior strength on mean (higher = more confident prior).
    prior_alpha
        Inverse-Gamma shape for variance prior.
    prior_beta
        Inverse-Gamma scale for variance prior.
    warmup
        Number of initial observations to use for prior calibration
        before scoring begins.
    anomaly_scale
        Scale multiplier for the anomaly hypothesis in Bayes factor.
    id_col
        Column identifying each time series.
    target_col
        Column with target values.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        threshold: float = 0.01,
        prior_mu: float | None = None,
        prior_kappa: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
        warmup: int = 10,
        anomaly_scale: float = 10.0,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if anomaly_scale <= 1:
            raise ValueError("anomaly_scale must be > 1")

        self.threshold = threshold
        self.prior_mu = prior_mu
        self.prior_kappa = prior_kappa
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.warmup = warmup
        self.anomaly_scale = anomaly_scale
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col

    def score(self, df: pl.DataFrame) -> BayesianAnomalyResult:
        """Score each observation for anomalousness.

        Parameters
        ----------
        df
            Input DataFrame with time series data.

        Returns
        -------
        BayesianAnomalyResult
            Result containing scores DataFrame and anomaly count.

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        all_rows: list[dict[str, Any]] = []

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = group_df[self.target_col].to_numpy().astype(np.float64)
            rows = self._score_single(gid, values)
            all_rows.extend(rows)

        scores_df = pl.DataFrame(all_rows)
        n_anomalies = int(scores_df["is_anomaly"].sum()) if len(scores_df) > 0 else 0

        return BayesianAnomalyResult(scores=scores_df, n_anomalies=n_anomalies)

    def _score_single(self, gid: Any, values: np.ndarray) -> list[dict[str, Any]]:
        """Score a single time series."""
        n = len(values)

        # Initialize prior
        mu0 = self.prior_mu if self.prior_mu is not None else (float(values[0]) if n > 0 else 0.0)
        state = _NIGState(
            mu=mu0,
            kappa=self.prior_kappa,
            alpha=self.prior_alpha,
            beta=self.prior_beta,
        )

        rows: list[dict[str, Any]] = []
        for t in range(n):
            y = float(values[t])

            if t < self.warmup:
                # Warmup phase: update prior, don't score
                p_value = 1.0
                bayes_factor = 1.0
                is_anomaly = False
            else:
                p_value = _compute_pvalue(y, state)
                bayes_factor = _compute_bayes_factor(y, state, self.anomaly_scale)
                is_anomaly = p_value < self.threshold

            rows.append(
                {
                    self.id_col: gid,
                    "t": t,
                    "value": y,
                    "p_value": p_value,
                    "bayes_factor": bayes_factor,
                    "is_anomaly": is_anomaly,
                }
            )

            state.update(y)

        return rows


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def bayesian_anomaly_score(
    df: pl.DataFrame,
    threshold: float = 0.01,
    prior_mu: float | None = None,
    prior_kappa: float = 1.0,
    prior_alpha: float = 2.0,
    prior_beta: float = 1.0,
    warmup: int = 10,
    anomaly_scale: float = 10.0,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Bayesian anomaly scoring convenience function.

    Scores each observation using an online conjugate normal model with
    posterior predictive p-values and Bayes factors.

    Parameters
    ----------
    df
        Input DataFrame.
    threshold
        P-value threshold for flagging anomalies (default 0.01).
    prior_mu
        Prior mean (default: first observation).
    prior_kappa
        Prior strength on mean.
    prior_alpha
        Inverse-Gamma shape for variance prior.
    prior_beta
        Inverse-Gamma scale for variance prior.
    warmup
        Warmup observations before scoring.
    anomaly_scale
        Scale for anomaly hypothesis in Bayes factor.
    id_col, target_col, time_col
        Column names.

    Returns
    -------
    pl.DataFrame
        Scores with ``p_value``, ``bayes_factor``, ``is_anomaly`` columns.

    """
    detector = BayesianAnomalyDetector(
        threshold=threshold,
        prior_mu=prior_mu,
        prior_kappa=prior_kappa,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        warmup=warmup,
        anomaly_scale=anomaly_scale,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
    )
    return detector.score(df).scores
