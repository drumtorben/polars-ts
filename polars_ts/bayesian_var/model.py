"""Bayesian Vector Autoregression (BVAR) model class and helpers."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from polars_ts.bayesian_var.inference import _analytical_posterior, _gibbs_sample
from polars_ts.bayesian_var.priors import (
    MinnesotaPrior,
    NormalWishartPrior,
    _estimate_sigma_from_ar,
    _minnesota_prior_precision,
)
from polars_ts.bayesian_var.results import BayesianVARResult

PriorType = Literal["minnesota", "normal_wishart"]
InferenceMethod = Literal["analytical", "gibbs"]


def _build_var_matrices(
    data: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build design matrix X and response Y for VAR(p).

    Parameters
    ----------
    data
        Array of shape ``(n, k)`` with the multivariate time series.
    p
        Number of lags.

    Returns
    -------
    X
        Design matrix, shape ``(n-p, k*p+1)`` (includes intercept).
    Y
        Response matrix, shape ``(n-p, k)``.

    """
    n, k = data.shape
    T = n - p
    X = np.empty((T, k * p + 1))
    Y = data[p:]

    for t in range(T):
        row = []
        for lag in range(1, p + 1):
            row.extend(data[p + t - lag])
        row.append(1.0)
        X[t] = row

    return X, Y


class BayesianVAR:
    """Bayesian Vector Autoregression forecaster.

    Parameters
    ----------
    target_cols
        Column names to model jointly (>= 2).
    p
        Number of lags.
    prior
        Prior type: ``"minnesota"`` or ``"normal_wishart"``.
    inference
        Inference method: ``"analytical"`` (conjugate posterior) or
        ``"gibbs"`` (Gibbs sampling).
    minnesota_prior
        Minnesota prior settings. Used when ``prior="minnesota"``.
    nw_prior
        Normal-Wishart prior settings. Used when ``prior="normal_wishart"``.
    coverage
        Credible interval coverage level (default 0.9).
    n_samples
        Number of Gibbs samples (after burn-in).
    burn_in
        Number of Gibbs burn-in samples.
    seed
        Random seed.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        target_cols: list[str],
        p: int = 1,
        prior: PriorType = "minnesota",
        inference: InferenceMethod = "analytical",
        minnesota_prior: MinnesotaPrior | None = None,
        nw_prior: NormalWishartPrior | None = None,
        coverage: float = 0.9,
        n_samples: int = 1000,
        burn_in: int = 500,
        seed: int = 42,
        time_col: str = "ds",
    ) -> None:
        if len(target_cols) < 2:
            raise ValueError("BVAR requires at least 2 target columns")
        if p < 1:
            raise ValueError("p must be >= 1")
        if prior not in ("minnesota", "normal_wishart"):
            raise ValueError(f"prior must be 'minnesota' or 'normal_wishart', got {prior!r}")
        if inference not in ("analytical", "gibbs"):
            raise ValueError(f"inference must be 'analytical' or 'gibbs', got {inference!r}")
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")

        self.target_cols = list(target_cols)
        self.p = p
        self.prior = prior
        self.inference = inference
        self.minnesota_prior = minnesota_prior or MinnesotaPrior()
        self.nw_prior = nw_prior or NormalWishartPrior()
        self.coverage = coverage
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.seed = seed
        self.time_col = time_col

        self._results: dict[Any, BayesianVARResult] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame, id_col: str | None = None) -> BayesianVAR:
        """Fit the Bayesian VAR model."""
        if id_col is not None:
            sorted_df = df.sort(id_col, self.time_col)
            for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
                gid = group_id[0]
                self._fit_single(group_df, gid)
        else:
            self._fit_single(df, "__global__")

        self.is_fitted_ = True
        return self

    def _fit_single(self, df: pl.DataFrame, gid: Any) -> None:
        """Fit BVAR on a single group/series."""
        sorted_df = df.sort(self.time_col)
        data = sorted_df.select(self.target_cols).to_numpy().astype(np.float64)
        n, k = data.shape

        if n <= self.p:
            raise ValueError(f"Need more than {self.p} observations for VAR({self.p}), got {n}")

        X, Y = _build_var_matrices(data, self.p)
        dim = k * self.p + 1

        if self.prior == "minnesota":
            sigma_scale = self.minnesota_prior.sigma_scale
            if sigma_scale is None:
                sigma_scale = _estimate_sigma_from_ar(data, self.p)
            B0, V0_inv_diag = _minnesota_prior_precision(k, self.p, self.minnesota_prior, sigma_scale)
        else:
            nw = self.nw_prior
            B0 = nw.B0 if nw.B0 is not None else np.zeros((k, dim))
            if nw.B0 is None:
                for j in range(k):
                    B0[j, j] = 1.0
            if nw.V0 is not None:
                V0_inv_diag = np.diag(nw.V0)
            else:
                V0_inv_diag = np.full(dim, 1.0 / max(nw.tightness**2, 1e-20))
                V0_inv_diag[-1] = 1e-6

        if self.prior == "normal_wishart" and self.nw_prior.S0 is not None:
            S0 = self.nw_prior.S0
        else:
            S0 = np.diag(_estimate_sigma_from_ar(data, self.p))

        nu0 = k + 2
        if self.prior == "normal_wishart" and self.nw_prior.nu0 is not None:
            nu0 = self.nw_prior.nu0

        if self.inference == "analytical":
            B_post, _V_post_inv, S_post, nu_post = _analytical_posterior(X, Y, B0, V0_inv_diag, S0, nu0)
            Sigma_post = S_post / max(nu_post - k - 1, 1)
            result = BayesianVARResult(
                B_post=B_post,
                Sigma_post=Sigma_post,
                target_cols=self.target_cols,
                p=self.p,
                _last_values=data[-self.p :],
            )
        else:
            B_samples, Sigma_samples = _gibbs_sample(
                X,
                Y,
                B0,
                V0_inv_diag,
                S0,
                nu0,
                self.n_samples,
                self.burn_in,
                self.seed,
            )
            B_post = np.mean(B_samples, axis=0)
            Sigma_post = np.mean(Sigma_samples, axis=0)
            result = BayesianVARResult(
                B_post=B_post,
                Sigma_post=Sigma_post,
                B_samples=B_samples,
                Sigma_samples=Sigma_samples,
                target_cols=self.target_cols,
                p=self.p,
                _last_values=data[-self.p :],
            )

        self._results[gid] = result

    def predict(self, horizon: int, id_col: str | None = None) -> pl.DataFrame:
        """Generate multi-step forecasts with credible intervals."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        alpha_half = (1 - self.coverage) / 2
        all_rows: list[dict[str, Any]] = []

        for gid, result in self._results.items():
            forecasts = self._forecast_single(result, horizon, alpha_half)
            for step_data in forecasts:
                row: dict[str, Any] = {}
                if id_col is not None and gid != "__global__":
                    row[id_col] = gid
                row[self.time_col] = step_data["step"]
                for col in self.target_cols:
                    row[col] = step_data[col]
                    row[f"{col}_lower"] = step_data[f"{col}_lower"]
                    row[f"{col}_upper"] = step_data[f"{col}_upper"]
                all_rows.append(row)

        return pl.DataFrame(all_rows)

    def _forecast_single(self, result: BayesianVARResult, horizon: int, alpha_half: float) -> list[dict[str, Any]]:
        """Forecast from a single fitted result."""
        k = len(self.target_cols)
        p = self.p

        if self.inference == "gibbs" and result.B_samples is not None and result.Sigma_samples is not None:
            n_samp = len(result.B_samples)
            rng = np.random.default_rng(self.seed)
            all_fc = np.empty((n_samp, horizon, k))

            for i in range(n_samp):
                B = result.B_samples[i]
                Sigma = result.Sigma_samples[i]
                history = list(result._last_values.copy())
                try:
                    L = np.linalg.cholesky(Sigma + np.eye(k) * 1e-10)
                except np.linalg.LinAlgError:
                    L = np.diag(np.sqrt(np.maximum(np.diag(Sigma), 1e-10)))
                for step in range(horizon):
                    x_parts: list[float] = []
                    for lag in range(1, p + 1):
                        x_parts.extend(history[len(history) - lag])
                    x_parts.append(1.0)
                    x = np.array(x_parts)
                    pred = B @ x + L @ rng.standard_normal(k)
                    all_fc[i, step] = pred
                    history.append(pred)

            mean_fc = np.mean(all_fc, axis=0)
            lower_fc = np.quantile(all_fc, alpha_half, axis=0)
            upper_fc = np.quantile(all_fc, 1 - alpha_half, axis=0)
        else:
            B = result.B_post
            history = list(result._last_values.copy())
            mean_fc = np.empty((horizon, k))
            for step in range(horizon):
                x_parts = []
                for lag in range(1, p + 1):
                    x_parts.extend(history[len(history) - lag])
                x_parts.append(1.0)
                x = np.array(x_parts)
                pred = B @ x
                mean_fc[step] = pred
                history.append(pred)

            from scipy.stats import norm

            z = norm.ppf(1 - alpha_half)
            sigma_diag = np.sqrt(np.maximum(np.diag(result.Sigma_post), 1e-10))
            horizon_scale = np.sqrt(np.arange(1, horizon + 1))[:, None]
            lower_fc = mean_fc - z * sigma_diag * horizon_scale
            upper_fc = mean_fc + z * sigma_diag * horizon_scale

        rows: list[dict[str, Any]] = []
        for step in range(horizon):
            row: dict[str, Any] = {"step": step + 1}
            for j, col in enumerate(self.target_cols):
                row[col] = float(mean_fc[step, j])
                row[f"{col}_lower"] = float(lower_fc[step, j])
                row[f"{col}_upper"] = float(upper_fc[step, j])
            rows.append(row)
        return rows

    def irf(self, steps: int = 20, shock_size: float = 1.0, gid: Any = None) -> pl.DataFrame:
        """Compute impulse response functions with credible bands."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before irf()")
        if steps <= 0:
            raise ValueError("steps must be positive")

        key = gid if gid is not None else "__global__"
        result = self._results.get(key)
        if result is None:
            raise ValueError(f"Group {gid!r} not found in fitted model")

        k = len(self.target_cols)
        p = self.p
        alpha_half = (1 - self.coverage) / 2

        if self.inference == "gibbs" and result.B_samples is not None:
            n_samp = len(result.B_samples)
            all_irfs = np.empty((n_samp, steps, k, k))
            for s in range(n_samp):
                all_irfs[s] = self._compute_irf(result.B_samples[s], k, p, steps, shock_size)
            mean_irf = np.mean(all_irfs, axis=0)
            lower_irf = np.quantile(all_irfs, alpha_half, axis=0)
            upper_irf = np.quantile(all_irfs, 1 - alpha_half, axis=0)
        else:
            mean_irf = self._compute_irf(result.B_post, k, p, steps, shock_size)
            from scipy.stats import norm

            z = norm.ppf(1 - alpha_half)
            sigma_scale = np.sqrt(np.maximum(np.diag(result.Sigma_post), 1e-10))
            lower_irf = np.empty_like(mean_irf)
            upper_irf = np.empty_like(mean_irf)
            for t in range(steps):
                scale = z * sigma_scale * np.sqrt(t + 1) * 0.1
                for i in range(k):
                    for j in range(k):
                        lower_irf[t, i, j] = mean_irf[t, i, j] - scale[j]
                        upper_irf[t, i, j] = mean_irf[t, i, j] + scale[j]

        rows: list[dict[str, Any]] = []
        for t in range(steps):
            for i in range(k):
                for j in range(k):
                    rows.append(
                        {
                            "step": t + 1,
                            "impulse": self.target_cols[i],
                            "response": self.target_cols[j],
                            "irf": float(mean_irf[t, i, j]),
                            "irf_lower": float(lower_irf[t, i, j]),
                            "irf_upper": float(upper_irf[t, i, j]),
                        }
                    )
        return pl.DataFrame(rows)

    @staticmethod
    def _compute_irf(B: np.ndarray, k: int, p: int, steps: int, shock_size: float) -> np.ndarray:
        """Compute orthogonalized IRF from coefficient matrix."""
        A_mats = [B[:, lag * k : (lag + 1) * k] for lag in range(p)]
        Phi = np.zeros((steps, k, k))
        Phi[0] = np.eye(k) * shock_size
        for t in range(1, steps):
            for lag in range(min(t, p)):
                Phi[t] += A_mats[lag] @ Phi[t - lag - 1]
        return Phi


def bayesian_var(
    df: pl.DataFrame,
    target_cols: list[str],
    horizon: int,
    p: int = 1,
    prior: PriorType = "minnesota",
    inference: InferenceMethod = "analytical",
    minnesota_prior: MinnesotaPrior | None = None,
    nw_prior: NormalWishartPrior | None = None,
    coverage: float = 0.9,
    n_samples: int = 1000,
    burn_in: int = 500,
    seed: int = 42,
    time_col: str = "ds",
    id_col: str | None = None,
) -> pl.DataFrame:
    """Bayesian VAR convenience function."""
    model = BayesianVAR(
        target_cols=target_cols,
        p=p,
        prior=prior,
        inference=inference,
        minnesota_prior=minnesota_prior,
        nw_prior=nw_prior,
        coverage=coverage,
        n_samples=n_samples,
        burn_in=burn_in,
        seed=seed,
        time_col=time_col,
    )
    model.fit(df, id_col=id_col)
    return model.predict(horizon=horizon, id_col=id_col)
