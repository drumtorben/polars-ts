"""Bayesian Exponential Smoothing (Bayesian ETS).

Places priors over smoothing parameters and initial states, producing
posterior predictive distributions instead of point forecasts.

Supports SES, Holt, and Holt-Winters (additive/multiplicative) with
MAP estimation via L-BFGS-B or full posterior via Metropolis-Hastings MCMC.

References
----------
- Hyndman et al. (2008), *Forecasting with Exponential Smoothing*
- Smyl (2020), *A hybrid method of exponential smoothing and recurrent
  neural networks for time series forecasting*

"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates

# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------

ModelType = Literal["ses", "holt", "holt_winters"]
InferenceMethod = Literal["map", "mcmc"]


@dataclass
class ETSPriors:
    """Prior distributions for ETS smoothing parameters and initial states.

    Smoothing parameters (alpha, beta, gamma) use Beta priors on (0, 1).
    Initial level/trend use Normal priors.

    Parameters
    ----------
    alpha_a, alpha_b
        Beta prior shape parameters for *alpha*.
    beta_a, beta_b
        Beta prior shape parameters for *beta* (Holt / Holt-Winters).
    gamma_a, gamma_b
        Beta prior shape parameters for *gamma* (Holt-Winters).
    level_mu, level_sigma
        Normal prior mean/std for initial level.
    trend_mu, trend_sigma
        Normal prior mean/std for initial trend.
    sigma_shape, sigma_scale
        Inverse-Gamma prior shape/scale for observation noise variance.

    """

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


# ---------------------------------------------------------------------------
# State-space log-likelihood
# ---------------------------------------------------------------------------


def _ses_loglik(
    values: list[float],
    alpha: float,
    level0: float,
    sigma: float,
) -> float:
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


def _holt_loglik(
    values: list[float],
    alpha: float,
    beta: float,
    level0: float,
    trend0: float,
    sigma: float,
) -> float:
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


# ---------------------------------------------------------------------------
# Log-prior
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------


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


def _unpack_params(
    theta: np.ndarray,
    model: ModelType,
    m: int,
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Log-posterior
# ---------------------------------------------------------------------------


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

    # Check bounds on smoothing parameters
    for key in ("alpha", "beta", "gamma"):
        val = p[key]
        if val is not None and (val <= 0 or val >= 1):
            return -np.inf
    if p["sigma"] <= 0:
        return -np.inf

    # Log-likelihood
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

    # Log-prior
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


# ---------------------------------------------------------------------------
# MAP estimation
# ---------------------------------------------------------------------------


def _map_estimate(
    values: list[float],
    model: ModelType,
    m: int,
    additive: bool,
    priors: ETSPriors,
) -> np.ndarray:
    """Find MAP estimate via L-BFGS-B."""
    # Sensible initial values
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

    # Bounds
    eps = 1e-6
    bounds: list[tuple[float | None, float | None]] = [(eps, 1 - eps)]  # alpha
    if model in ("holt", "holt_winters"):
        bounds.append((eps, 1 - eps))  # beta
    if model == "holt_winters":
        bounds.append((eps, 1 - eps))  # gamma
    bounds.append((None, None))  # level0
    if model in ("holt", "holt_winters"):
        bounds.append((None, None))  # trend0
    if model == "holt_winters":
        bounds.extend([(None, None)] * m)  # seasons
    bounds.append((eps, None))  # sigma

    def neg_log_post(theta: np.ndarray) -> float:
        val = _log_posterior(theta, values, model, m, additive, priors)
        return -val if np.isfinite(val) else 1e20

    from scipy.optimize import minimize

    result = minimize(neg_log_post, x0, method="L-BFGS-B", bounds=bounds)
    return result.x


# ---------------------------------------------------------------------------
# MCMC (Metropolis-Hastings)
# ---------------------------------------------------------------------------


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
    """Draw posterior samples via Metropolis-Hastings.

    Returns array of shape ``(n_samples, n_params)``.
    """
    rng = np.random.default_rng(seed)

    # Start from MAP
    theta_current = _map_estimate(values, model, m, additive, priors)
    n_params = len(theta_current)
    lp_current = _log_posterior(theta_current, values, model, m, additive, priors)

    # Adaptive proposal scale
    proposal_scale = np.abs(theta_current) * 0.01
    proposal_scale = np.maximum(proposal_scale, 1e-4)

    total = n_samples + burn_in
    samples = np.empty((total, n_params))
    accepted = 0

    for i in range(total):
        theta_proposal = theta_current + rng.normal(0, proposal_scale)
        lp_proposal = _log_posterior(theta_proposal, values, model, m, additive, priors)

        log_ratio = lp_proposal - lp_current
        if np.isfinite(log_ratio) and math.log(rng.uniform()) < log_ratio:
            theta_current = theta_proposal
            lp_current = lp_proposal
            accepted += 1

        samples[i] = theta_current

    return samples[burn_in:]


# ---------------------------------------------------------------------------
# Forecasting from parameters
# ---------------------------------------------------------------------------


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

    # Optionally add observation noise for posterior predictive samples
    if sigma_noise and rng is not None:
        sigma = params["sigma"]
        forecasts = [f + rng.normal(0, sigma) for f in forecasts]

    return forecasts


# ---------------------------------------------------------------------------
# BayesianETS class
# ---------------------------------------------------------------------------


@dataclass
class BayesianETSResult:
    """Fitted result from BayesianETS.

    Attributes
    ----------
    map_params
        MAP parameter estimates (dict).
    posterior_samples
        MCMC posterior samples, shape ``(n_samples, n_params)``.
        ``None`` for MAP inference.

    """

    map_params: dict[str, Any]
    posterior_samples: np.ndarray | None = None


class BayesianETS:
    """Bayesian Exponential Smoothing forecaster.

    Places priors over smoothing parameters and initial states, producing
    posterior predictive forecasts with credible intervals.

    Parameters
    ----------
    model
        Model type: ``"ses"``, ``"holt"``, or ``"holt_winters"``.
    inference
        Inference method: ``"map"`` (fast) or ``"mcmc"`` (full posterior).
    season_length
        Number of observations per season (required for ``"holt_winters"``).
    seasonal
        ``"additive"`` or ``"multiplicative"`` (Holt-Winters only).
    priors
        Prior specification. Uses default ``ETSPriors()`` if not provided.
    coverage
        Credible interval coverage level (e.g. 0.9 for 90%).
    n_samples
        Number of MCMC samples to draw (after burn-in).
    burn_in
        Number of MCMC burn-in samples to discard.
    seed
        Random seed for MCMC.
    target_col
        Column with target values.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        model: ModelType = "ses",
        inference: InferenceMethod = "map",
        season_length: int = 1,
        seasonal: str = "additive",
        priors: ETSPriors | None = None,
        coverage: float = 0.9,
        n_samples: int = 1000,
        burn_in: int = 500,
        seed: int = 42,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        if model not in ("ses", "holt", "holt_winters"):
            raise ValueError(f"model must be 'ses', 'holt', or 'holt_winters', got {model!r}")
        if inference not in ("map", "mcmc"):
            raise ValueError(f"inference must be 'map' or 'mcmc', got {inference!r}")
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        if model == "holt_winters" and season_length < 2:
            raise ValueError("season_length must be >= 2 for holt_winters")
        if seasonal not in ("additive", "multiplicative"):
            raise ValueError(f"seasonal must be 'additive' or 'multiplicative', got {seasonal!r}")

        self.model = model
        self.inference = inference
        self.season_length = season_length
        self.seasonal = seasonal
        self.additive = seasonal == "additive"
        self.priors = priors or ETSPriors()
        self.coverage = coverage
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.seed = seed
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col

        self._results: dict[Any, BayesianETSResult] = {}
        self._group_values: dict[Any, list[float]] = {}
        self.is_fitted_: bool = False

    @property
    def _m(self) -> int:
        return self.season_length if self.model == "holt_winters" else 1

    def fit(self, df: pl.DataFrame) -> BayesianETS:
        """Fit the Bayesian ETS model.

        Parameters
        ----------
        df
            Input DataFrame with time series data.

        Returns
        -------
        BayesianETS
            Fitted instance (``self``).

        """
        sorted_df = df.sort(self.id_col, self.time_col)
        m = self._m

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = [float(v) for v in group_df[self.target_col].to_list()]

            if self.model == "holt" and len(values) < 2:
                raise ValueError(f"Series {gid!r} needs at least 2 observations for Holt's method")
            if self.model == "holt_winters" and len(values) < 2 * m:
                raise ValueError(
                    f"Series {gid!r} needs at least 2*season_length={2 * m} observations, " f"got {len(values)}"
                )

            # Center level prior on data mean
            priors = replace(self.priors, level_mu=float(np.mean(values)))

            # MAP estimate
            map_theta = _map_estimate(values, self.model, m, self.additive, priors)
            map_params = _unpack_params(map_theta, self.model, m)

            # MCMC samples
            posterior_samples = None
            if self.inference == "mcmc":
                posterior_samples = _mcmc_sample(
                    values,
                    self.model,
                    m,
                    self.additive,
                    priors,
                    n_samples=self.n_samples,
                    burn_in=self.burn_in,
                    seed=self.seed,
                )

            self._results[gid] = BayesianETSResult(
                map_params=map_params,
                posterior_samples=posterior_samples,
            )
            self._group_values[gid] = values

        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts with credible intervals.

        Parameters
        ----------
        df
            DataFrame containing history to predict from.
        h
            Forecast horizon.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, time_col, "y_hat",
            "y_hat_lower", "y_hat_upper"]``.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("Horizon h must be a positive integer")

        sorted_df = df.sort(self.id_col, self.time_col)
        freq = _infer_freq(sorted_df[self.time_col])
        m = self._m
        alpha_half = (1 - self.coverage) / 2

        rows: list[dict[str, Any]] = []
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            values = [float(v) for v in group_df[self.target_col].to_list()]
            last_time = group_df[self.time_col][-1]
            future_times = _make_future_dates(last_time, freq, h)

            result = self._results.get(gid)
            if result is None:
                raise ValueError(f"Series {gid!r} was not seen during fit()")

            if self.inference == "mcmc" and result.posterior_samples is not None:
                # Posterior predictive: forecast from each MCMC sample
                rng = np.random.default_rng(self.seed)
                all_forecasts = np.empty((len(result.posterior_samples), h))
                for i, sample in enumerate(result.posterior_samples):
                    params = _unpack_params(sample, self.model, m)
                    all_forecasts[i] = _forecast_from_params(
                        values,
                        params,
                        self.model,
                        m,
                        self.additive,
                        h,
                        sigma_noise=True,
                        rng=rng,
                    )

                y_hat = np.mean(all_forecasts, axis=0)
                y_lower = np.quantile(all_forecasts, alpha_half, axis=0)
                y_upper = np.quantile(all_forecasts, 1 - alpha_half, axis=0)
            else:
                # MAP: point forecast + approximate intervals from residual variance
                y_hat_list = _forecast_from_params(
                    values,
                    result.map_params,
                    self.model,
                    m,
                    self.additive,
                    h,
                )
                from scipy.stats import norm

                sigma = result.map_params["sigma"]
                z = norm.ppf(1 - alpha_half)
                # Uncertainty grows with horizon
                y_hat = np.array(y_hat_list)
                y_lower = y_hat - z * sigma * np.sqrt(np.arange(1, h + 1))
                y_upper = y_hat + z * sigma * np.sqrt(np.arange(1, h + 1))

            for step in range(h):
                rows.append(
                    {
                        self.id_col: gid,
                        self.time_col: future_times[step],
                        "y_hat": float(y_hat[step]),
                        "y_hat_lower": float(y_lower[step]),
                        "y_hat_upper": float(y_upper[step]),
                    }
                )

        schema: dict[str, Any] = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
            "y_hat": pl.Float64(),
            "y_hat_lower": pl.Float64(),
            "y_hat_upper": pl.Float64(),
        }
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def bayesian_ets(
    df: pl.DataFrame,
    h: int,
    model: ModelType = "ses",
    inference: InferenceMethod = "map",
    season_length: int = 1,
    seasonal: str = "additive",
    priors: ETSPriors | None = None,
    coverage: float = 0.9,
    n_samples: int = 1000,
    burn_in: int = 500,
    seed: int = 42,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Bayesian ETS convenience function.

    Fits a Bayesian ETS model and returns posterior predictive forecasts
    with credible intervals in a single call.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    model
        ``"ses"``, ``"holt"``, or ``"holt_winters"``.
    inference
        ``"map"`` or ``"mcmc"``.
    season_length
        Observations per season (Holt-Winters).
    seasonal
        ``"additive"`` or ``"multiplicative"``.
    priors
        Prior specification.
    coverage
        Credible interval coverage (default 0.9).
    n_samples
        MCMC samples (after burn-in).
    burn_in
        MCMC burn-in samples.
    seed
        Random seed.
    target_col, id_col, time_col
        Column names.

    Returns
    -------
    pl.DataFrame
        Forecasts with ``y_hat``, ``y_hat_lower``, ``y_hat_upper``.

    """
    estimator = BayesianETS(
        model=model,
        inference=inference,
        season_length=season_length,
        seasonal=seasonal,
        priors=priors,
        coverage=coverage,
        n_samples=n_samples,
        burn_in=burn_in,
        seed=seed,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
    )
    estimator.fit(df)
    return estimator.predict(df, h)
