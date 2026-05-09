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

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from polars_ts.bayesian.mcmc_backends import _run_numpyro, _run_pymc
from polars_ts.bayesian.mcmc_forecast import _forecast_ar, _forecast_local_level, _forecast_seasonal
from polars_ts.bayesian.mcmc_samplers import _ar_logpost, _local_level_logpost, _mh_sample, _seasonal_logpost

ModelType = Literal["local_level", "ar", "seasonal"]
BackendType = Literal["builtin", "numpyro", "pymc"]


@dataclass
class MCMCResult:
    """Container for MCMC sampling results."""

    samples: dict[str, np.ndarray]
    forecast: np.ndarray | None = None
    point_forecast: np.ndarray = field(default_factory=lambda: np.empty(0))
    lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper: np.ndarray = field(default_factory=lambda: np.empty(0))


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
            samples = _run_numpyro(y, self.model, self.n_samples, self.burn_in, self.seed, p=self.p)
        else:
            samples = _run_pymc(y, self.model, self.n_samples, self.burn_in, self.seed, p=self.p)

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

    def _forecast_builtin(self, y: np.ndarray, samples: dict[str, np.ndarray], h: int) -> np.ndarray:
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
    """MCMC forecasting convenience function."""
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
