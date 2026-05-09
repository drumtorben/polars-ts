"""External MCMC backends (NumPyro, PyMC) for MCMCForecaster."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

ModelType = Literal["local_level", "ar", "seasonal"]


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
