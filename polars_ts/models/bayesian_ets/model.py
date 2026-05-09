"""BayesianETS model class, result container, and convenience function."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates
from polars_ts.models.bayesian_ets.inference import (
    _forecast_from_params,
    _map_estimate,
    _mcmc_sample,
    _unpack_params,
)
from polars_ts.models.bayesian_ets.priors import ETSPriors, InferenceMethod, ModelType


@dataclass
class BayesianETSResult:
    """Fitted result from BayesianETS."""

    map_params: dict[str, Any]
    posterior_samples: np.ndarray | None = None


class BayesianETS:
    """Bayesian Exponential Smoothing forecaster."""

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
        """Fit the Bayesian ETS model."""
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

            priors = replace(self.priors, level_mu=float(np.mean(values)))
            map_theta = _map_estimate(values, self.model, m, self.additive, priors)
            map_params = _unpack_params(map_theta, self.model, m)

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

            self._results[gid] = BayesianETSResult(map_params=map_params, posterior_samples=posterior_samples)
            self._group_values[gid] = values

        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts with credible intervals."""
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
                y_hat_list = _forecast_from_params(values, result.map_params, self.model, m, self.additive, h)
                from scipy.stats import norm

                sigma = result.map_params["sigma"]
                z = norm.ppf(1 - alpha_half)
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
    """Bayesian ETS convenience function."""
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
