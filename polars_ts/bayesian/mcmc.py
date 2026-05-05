"""MCMC forecasting wrapper for time series models.

Provides adapter layers around NumPyro and PyMC for posterior sampling
of time series models, plus a built-in lightweight Metropolis-Hastings
sampler that works without external PPL dependencies.

Built-in models: local level, AR(p), seasonal local level.

References
----------
- Phan et al. (2019), *Composable Effects for Flexible and Accelerated
  Probabilistic Programming in NumPyro*
- Abril-Pla et al. (2023), *PyMC: a modern, and comprehensive
  probabilistic programming framework in Python*

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

ModelType = Literal["local_level", "ar", "seasonal"]
BackendType = Literal["builtin", "numpyro", "pymc"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult:
    """Container for MCMC sampling results.

    Attributes
    ----------
    samples
        Dict mapping parameter name to array of posterior samples.
    forecast
        Posterior predictive forecast, shape ``(n_samples, h)``.
        ``None`` if no forecast was requested.
    point_forecast
        Posterior mean forecast, shape ``(h,)``.
    lower
        Lower credible bound, shape ``(h,)``.
    upper
        Upper credible bound, shape ``(h,)``.

    """

    samples: dict[str, np.ndarray]
    forecast: np.ndarray | None = None
    point_forecast: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper: np.ndarray = field(default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# Built-in log-posteriors
# ---------------------------------------------------------------------------


def _local_level_logpost(
    params: np.ndarray,
    y: np.ndarray,
) -> float:
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
        # Use filtered update: level ~ N(alpha * y_t + (1-alpha) * level, ...)
        alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2)
        level = alpha * y[t] + (1 - alpha) * level

    # Priors: half-normal on sigmas, normal on level0
    lp = -0.5 * (level0 / 100.0) ** 2
    lp += -0.5 * (sigma_obs / 10.0) ** 2
    lp += -0.5 * (sigma_level / 10.0) ** 2

    return ll + lp


def _ar_logpost(
    params: np.ndarray,
    y: np.ndarray,
    p: int,
) -> float:
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

    # Priors
    lp = -0.5 * (sigma / 10.0) ** 2
    lp += -0.5 * (mu / 100.0) ** 2
    for j in range(p):
        lp += -0.5 * phi[j] ** 2  # N(0,1) prior on AR coefficients

    return ll + lp


def _seasonal_logpost(
    params: np.ndarray,
    y: np.ndarray,
    season_length: int,
) -> float:
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
        # Filtered updates
        alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2)
        level = alpha * (y[t] - s[s_idx]) + (1 - alpha) * level
        gamma = sigma_season**2 / (sigma_season**2 + sigma_obs**2)
        s[s_idx] = gamma * (y[t] - level) + (1 - gamma) * s[s_idx]

    # Priors
    lp = -0.5 * (level0 / 100.0) ** 2
    lp += -0.5 * (sigma_obs / 10.0) ** 2
    lp += -0.5 * (sigma_level / 10.0) ** 2
    lp += -0.5 * (sigma_season / 10.0) ** 2
    for si in seasons:
        lp += -0.5 * (si / 10.0) ** 2

    return ll + lp


# ---------------------------------------------------------------------------
# Built-in Metropolis-Hastings sampler
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forecast from posterior samples
# ---------------------------------------------------------------------------


def _forecast_local_level(
    y: np.ndarray,
    samples: np.ndarray,
    h: int,
    seed: int,
) -> np.ndarray:
    """Posterior predictive forecast for local level model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma_obs = abs(samples[i, 0])
        sigma_level = abs(samples[i, 1])
        level = samples[i, 2]

        # Filter through observations
        for t in range(len(y)):
            alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2 + 1e-20)
            level = alpha * y[t] + (1 - alpha) * level

        # Forecast
        for step in range(h):
            level += rng.normal(0, sigma_level)
            forecasts[i, step] = level + rng.normal(0, sigma_obs)

    return forecasts


def _forecast_ar(
    y: np.ndarray,
    samples: np.ndarray,
    h: int,
    p: int,
    seed: int,
) -> np.ndarray:
    """Posterior predictive forecast for AR(p) model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma = abs(samples[i, 0])
        mu = samples[i, 1]
        phi = samples[i, 2 : 2 + p]

        history = list(y[-p:])
        for step in range(h):
            pred = mu
            for j in range(p):
                pred += phi[j] * (history[-(j + 1)] - mu)
            pred += rng.normal(0, sigma)
            forecasts[i, step] = pred
            history.append(pred)

    return forecasts


def _forecast_seasonal(
    y: np.ndarray,
    samples: np.ndarray,
    h: int,
    season_length: int,
    seed: int,
) -> np.ndarray:
    """Posterior predictive forecast for seasonal local level model."""
    rng = np.random.default_rng(seed)
    n_samp = len(samples)
    n = len(y)
    forecasts = np.empty((n_samp, h))

    for i in range(n_samp):
        sigma_obs = abs(samples[i, 0])
        sigma_level = abs(samples[i, 1])
        sigma_season = abs(samples[i, 2])
        level = samples[i, 3]
        seasons = list(samples[i, 4 : 4 + season_length])

        # Filter
        for t in range(n):
            s_idx = t % season_length
            alpha = sigma_level**2 / (sigma_level**2 + sigma_obs**2 + 1e-20)
            level = alpha * (y[t] - seasons[s_idx]) + (1 - alpha) * level
            gamma = sigma_season**2 / (sigma_season**2 + sigma_obs**2 + 1e-20)
            seasons[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasons[s_idx]

        # Forecast
        for step in range(h):
            s_idx = (n + step) % season_length
            level += rng.normal(0, sigma_level)
            seasons[s_idx] += rng.normal(0, sigma_season)
            forecasts[i, step] = level + seasons[s_idx] + rng.normal(0, sigma_obs)

    return forecasts


# ---------------------------------------------------------------------------
# NumPyro backend
# ---------------------------------------------------------------------------


def _run_numpyro(
    y: np.ndarray,
    model_type: ModelType,
    n_samples: int,
    burn_in: int,
    seed: int,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Run MCMC via NumPyro NUTS."""
    try:
        import jax.numpy as jnp  # noqa: F401
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError:
        raise ImportError(
            "numpyro and jax are required for the numpyro backend. " "Install with: pip install numpyro jax jaxlib"
        ) from None

    import jax

    def local_level_model(y_obs: Any = None, n: int = 0) -> None:
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(10.0))
        sigma_level = numpyro.sample("sigma_level", dist.HalfNormal(10.0))
        level = numpyro.sample("level0", dist.Normal(0, 100.0))

        for t in range(n):
            level = numpyro.sample(f"level_{t}", dist.Normal(level, sigma_level))
            numpyro.sample(f"y_{t}", dist.Normal(level, sigma_obs), obs=y_obs[t] if y_obs is not None else None)

    def ar_model(y_obs: Any = None, n: int = 0, p: int = 1) -> None:
        sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
        mu = numpyro.sample("mu", dist.Normal(0, 100.0))
        phi = numpyro.sample("phi", dist.Normal(jnp.zeros(p), jnp.ones(p)))

        for t in range(p, n):
            pred = mu
            for j in range(p):
                pred = pred + phi[j] * (y_obs[t - j - 1] - mu)
            numpyro.sample(f"y_{t}", dist.Normal(pred, sigma), obs=y_obs[t] if y_obs is not None else None)

    if model_type == "local_level":
        model_fn = local_level_model
        model_args = {"y_obs": jax.numpy.array(y), "n": len(y)}
    elif model_type == "ar":
        p = kwargs.get("p", 1)
        model_fn = ar_model
        model_args = {"y_obs": jax.numpy.array(y), "n": len(y), "p": p}
    else:
        raise ValueError(f"NumPyro backend does not support model {model_type!r}")

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=burn_in, num_samples=n_samples)
    mcmc.run(jax.random.PRNGKey(seed), **model_args)
    return {k: np.array(v) for k, v in mcmc.get_samples().items()}


# ---------------------------------------------------------------------------
# PyMC backend
# ---------------------------------------------------------------------------


def _run_pymc(
    y: np.ndarray,
    model_type: ModelType,
    n_samples: int,
    burn_in: int,
    seed: int,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Run MCMC via PyMC NUTS."""
    try:
        import pymc as pm
    except ImportError:
        raise ImportError("pymc is required for the pymc backend. Install with: pip install pymc") from None

    if model_type == "ar":
        p = kwargs.get("p", 1)
        with pm.Model():
            sigma = pm.HalfNormal("sigma", sigma=10.0)
            mu = pm.Normal("mu", mu=0, sigma=100.0)
            phi = pm.Normal("phi", mu=0, sigma=1.0, shape=p)
            pm.AR("y", rho=phi, sigma=sigma, constant=True, init_dist=pm.Normal.dist(mu, 10.0), observed=y)
            trace = pm.sample(n_samples, tune=burn_in, random_seed=seed, progressbar=False)
        return {
            var: trace.posterior[var].values.reshape(-1, *trace.posterior[var].values.shape[2:])
            for var in ["sigma", "mu", "phi"]
        }

    raise ValueError(f"PyMC backend does not support model {model_type!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MCMCForecaster:
    """MCMC-based time series forecaster.

    Parameters
    ----------
    model
        Model type: ``"local_level"``, ``"ar"``, or ``"seasonal"``.
    backend
        MCMC backend: ``"builtin"`` (no deps), ``"numpyro"``, or ``"pymc"``.
    p
        AR order (only for ``model="ar"``).
    season_length
        Season length (only for ``model="seasonal"``).
    coverage
        Credible interval coverage (default 0.9).
    n_samples
        Number of posterior samples.
    burn_in
        Number of warmup/burn-in samples.
    seed
        Random seed.
    id_col
        Column identifying each time series.
    target_col
        Column with target values.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        model: ModelType = "local_level",
        backend: BackendType = "builtin",
        p: int = 1,
        season_length: int = 12,
        coverage: float = 0.9,
        n_samples: int = 1000,
        burn_in: int = 500,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        if model not in ("local_level", "ar", "seasonal"):
            raise ValueError(f"model must be 'local_level', 'ar', or 'seasonal', got {model!r}")
        if backend not in ("builtin", "numpyro", "pymc"):
            raise ValueError(f"backend must be 'builtin', 'numpyro', or 'pymc', got {backend!r}")
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        if model == "ar" and p < 1:
            raise ValueError("p must be >= 1 for AR model")
        if model == "seasonal" and season_length < 2:
            raise ValueError("season_length must be >= 2 for seasonal model")

        self.model = model
        self.backend = backend
        self.p = p
        self.season_length = season_length
        self.coverage = coverage
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.seed = seed
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col

        self._results: dict[Any, MCMCResult] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> MCMCForecaster:
        """Fit the MCMC model to one or more time series."""
        sorted_df = df.sort(self.id_col, self.time_col)

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            y = group_df[self.target_col].to_numpy().astype(np.float64)
            self._fit_single(y, gid)

        self.is_fitted_ = True
        return self

    def _fit_single(self, y: np.ndarray, gid: Any) -> None:
        """Fit MCMC on a single series."""
        if self.backend == "builtin":
            samples = self._fit_builtin(y)
        elif self.backend == "numpyro":
            raw = _run_numpyro(y, self.model, self.n_samples, self.burn_in, self.seed, p=self.p)
            samples = raw
        else:
            raw = _run_pymc(y, self.model, self.n_samples, self.burn_in, self.seed, p=self.p)
            samples = raw

        self._results[gid] = MCMCResult(samples=samples)

    def _fit_builtin(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Run built-in MH sampler."""
        if self.model == "local_level":
            x0 = np.array([float(np.std(y)) or 1.0, 0.1, float(np.mean(y))])
            logpost = lambda params: _local_level_logpost(params, y)  # noqa: E731
            raw = _mh_sample(logpost, x0, self.n_samples, self.burn_in, self.seed)
            return {"sigma_obs": raw[:, 0], "sigma_level": raw[:, 1], "level0": raw[:, 2]}

        if self.model == "ar":
            x0 = np.zeros(2 + self.p)
            x0[0] = float(np.std(y)) or 1.0
            x0[1] = float(np.mean(y))
            logpost = lambda params: _ar_logpost(params, y, self.p)  # noqa: E731
            raw = _mh_sample(logpost, x0, self.n_samples, self.burn_in, self.seed)
            result = {"sigma": raw[:, 0], "mu": raw[:, 1]}
            for j in range(self.p):
                result[f"phi_{j+1}"] = raw[:, 2 + j]
            return result

        # seasonal
        m = self.season_length
        x0 = np.zeros(4 + m)
        x0[0] = float(np.std(y)) or 1.0
        x0[1] = 0.1
        x0[2] = 0.1
        x0[3] = float(np.mean(y))
        logpost = lambda params: _seasonal_logpost(params, y, m)  # noqa: E731
        raw = _mh_sample(logpost, x0, self.n_samples, self.burn_in, self.seed)
        result = {
            "sigma_obs": raw[:, 0],
            "sigma_level": raw[:, 1],
            "sigma_season": raw[:, 2],
            "level0": raw[:, 3],
        }
        for j in range(m):
            result[f"season_{j}"] = raw[:, 4 + j]
        return result

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step forecasts with credible intervals."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("h must be positive")

        alpha_half = (1 - self.coverage) / 2
        sorted_df = df.sort(self.id_col, self.time_col)
        all_rows: list[dict[str, Any]] = []

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            y = group_df[self.target_col].to_numpy().astype(np.float64)

            result = self._results.get(gid)
            if result is None:
                raise ValueError(f"Series {gid!r} was not seen during fit()")

            fc = self._forecast_builtin(y, result.samples, h)
            y_hat = np.mean(fc, axis=0)
            y_lower = np.quantile(fc, alpha_half, axis=0)
            y_upper = np.quantile(fc, 1 - alpha_half, axis=0)

            result.forecast = fc
            result.point_forecast = y_hat
            result.lower = y_lower
            result.upper = y_upper

            for step in range(h):
                all_rows.append(
                    {
                        self.id_col: gid,
                        "step": step + 1,
                        "y_hat": float(y_hat[step]),
                        "y_hat_lower": float(y_lower[step]),
                        "y_hat_upper": float(y_upper[step]),
                    }
                )

        return pl.DataFrame(all_rows)

    def _forecast_builtin(
        self,
        y: np.ndarray,
        samples: dict[str, np.ndarray],
        h: int,
    ) -> np.ndarray:
        """Generate posterior predictive forecasts from builtin samples."""
        if self.model == "local_level":
            raw = np.column_stack([samples["sigma_obs"], samples["sigma_level"], samples["level0"]])
            return _forecast_local_level(y, raw, h, self.seed)

        if self.model == "ar":
            cols = [samples["sigma"], samples["mu"]]
            for j in range(self.p):
                cols.append(samples[f"phi_{j+1}"])
            raw = np.column_stack(cols)
            return _forecast_ar(y, raw, h, self.p, self.seed)

        # seasonal
        m = self.season_length
        cols = [samples["sigma_obs"], samples["sigma_level"], samples["sigma_season"], samples["level0"]]
        for j in range(m):
            cols.append(samples[f"season_{j}"])
        raw = np.column_stack(cols)
        return _forecast_seasonal(y, raw, h, m, self.seed)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def mcmc_forecast(
    df: pl.DataFrame,
    h: int,
    model: ModelType = "local_level",
    backend: BackendType = "builtin",
    p: int = 1,
    season_length: int = 12,
    coverage: float = 0.9,
    n_samples: int = 1000,
    burn_in: int = 500,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
) -> pl.DataFrame:
    """MCMC forecasting convenience function.

    Fits a Bayesian time series model via MCMC and returns posterior
    predictive forecasts with credible intervals.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    model
        ``"local_level"``, ``"ar"``, or ``"seasonal"``.
    backend
        ``"builtin"`` (no extra deps), ``"numpyro"``, or ``"pymc"``.
    p
        AR order (for ``model="ar"``).
    season_length
        Season length (for ``model="seasonal"``).
    coverage
        Credible interval coverage (default 0.9).
    n_samples
        Number of posterior samples.
    burn_in
        Warmup/burn-in samples.
    seed
        Random seed.
    id_col, target_col, time_col
        Column names.

    Returns
    -------
    pl.DataFrame
        Forecasts with ``y_hat``, ``y_hat_lower``, ``y_hat_upper``.

    """
    forecaster = MCMCForecaster(
        model=model,
        backend=backend,
        p=p,
        season_length=season_length,
        coverage=coverage,
        n_samples=n_samples,
        burn_in=burn_in,
        seed=seed,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
    )
    forecaster.fit(df)
    return forecaster.predict(df, h)
