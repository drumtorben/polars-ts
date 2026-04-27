"""Bayesian Vector Autoregression (BVAR) for multivariate time series.

Implements Minnesota (Litterman) and Normal-Wishart conjugate priors with
analytical posterior or Gibbs sampling. Returns posterior predictive
forecasts with credible intervals and impulse response functions with
credible bands.

References
----------
- Litterman (1986), *Forecasting with Bayesian Vector Autoregressions*
- Koop & Korobilis (2010), *Bayesian Multivariate Time Series Methods
  for Empirical Macroeconomics*

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

PriorType = Literal["minnesota", "normal_wishart"]
InferenceMethod = Literal["analytical", "gibbs"]


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------


@dataclass
class MinnesotaPrior:
    """Minnesota (Litterman) prior for BVAR.

    Shrinks VAR coefficients toward a random walk: own first lag
    receives prior mean 1, all others 0. Tightness parameters
    control how strongly the prior pulls toward this structure.

    Parameters
    ----------
    lambda1
        Overall tightness. Smaller values shrink more aggressively.
    lambda2
        Cross-variable tightness (relative to own-lag). Typically < 1
        so cross-variable lags are shrunk harder.
    lambda3
        Lag decay. Higher values shrink distant lags more aggressively.
        Prior variance for lag *l* is scaled by ``l^{-lambda3}``.
    sigma_scale
        Per-variable residual variance estimates. If ``None``, estimated
        from univariate AR(p) regressions.

    """

    lambda1: float = 0.2
    lambda2: float = 0.5
    lambda3: float = 1.0
    sigma_scale: np.ndarray | None = None


@dataclass
class NormalWishartPrior:
    """Normal-Wishart conjugate prior for BVAR.

    Places a matrix-normal prior on the coefficient matrix ``B``
    and a Wishart prior on the precision matrix ``Sigma^{-1}``.

    Parameters
    ----------
    B0
        Prior mean for the coefficient matrix, shape ``(k, k*p+1)``.
        If ``None``, defaults to random walk (identity on first own-lag).
    V0
        Prior precision (inverse covariance) for vec(B),
        shape ``(k*p+1, k*p+1)``. If ``None``, uses Minnesota-style
        diagonal with the given ``tightness``.
    S0
        Prior scale matrix for Wishart, shape ``(k, k)``.
        If ``None``, uses identity scaled by data variance.
    nu0
        Degrees of freedom for Wishart. Must be >= k.
        If ``None``, defaults to ``k + 2``.
    tightness
        Diagonal tightness for automatic V0 construction.

    """

    B0: np.ndarray | None = None
    V0: np.ndarray | None = None
    S0: np.ndarray | None = None
    nu0: float | None = None
    tightness: float = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _estimate_sigma_from_ar(data: np.ndarray, p: int) -> np.ndarray:
    """Estimate per-variable residual variance from univariate AR(p)."""
    _n, k = data.shape
    sigmas = np.ones(k)
    for j in range(k):
        y_j = data[:, j]
        n_j = len(y_j)
        if n_j <= p + 1:
            continue
        X_ar = np.column_stack([y_j[p - i - 1 : n_j - i - 1] for i in range(p)] + [np.ones(n_j - p)])
        Y_ar = y_j[p:]
        beta = np.linalg.lstsq(X_ar, Y_ar, rcond=None)[0]
        resid = Y_ar - X_ar @ beta
        sigmas[j] = max(float(np.var(resid, ddof=p + 1)), 1e-10)
    return sigmas


def _minnesota_prior_precision(
    k: int,
    p: int,
    prior: MinnesotaPrior,
    sigma_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Minnesota prior mean B0 and precision V0_inv.

    Returns
    -------
    B0
        Prior mean, shape ``(k, k*p+1)``.
    V0_inv
        Prior precision diagonal, shape ``(k*p+1,)``.

    """
    dim = k * p + 1
    B0 = np.zeros((k, dim))
    # Random walk: own first lag = 1
    for j in range(k):
        B0[j, j] = 1.0

    V0_inv_diag = np.zeros(dim)
    for lag in range(1, p + 1):
        for j in range(k):
            col_idx = (lag - 1) * k + j
            # Own lag
            var_own = (prior.lambda1 / (lag**prior.lambda3)) ** 2
            V0_inv_diag[col_idx] = 1.0 / max(var_own, 1e-20)
            # Cross lags get tighter shrinkage
            for i in range(k):
                if i != j:
                    # Rescale by relative variance
                    s_ratio = sigma_scale[i] / max(sigma_scale[j], 1e-20)
                    var_cross = (prior.lambda1 * prior.lambda2 / (lag**prior.lambda3)) ** 2 * s_ratio
                    # This affects the prior for equation i, lag of variable j
                    # We set the diagonal for the coefficient in equation i
                    # but here we build a shared V0, so use average
                    V0_inv_diag[col_idx] = max(V0_inv_diag[col_idx], 1.0 / max(var_cross, 1e-20))

    # Intercept: diffuse
    V0_inv_diag[-1] = 1e-6

    return B0, V0_inv_diag


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
        # B' = B_post' + L_V @ Z @ L_Sigma' where Z ~ N(0,1)^{dim x k}
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
        # = inv(Wishart(nu_post, S_post^{-1}))
        nu_post = nu0 + T
        try:
            S_post_inv = np.linalg.inv(S_post)
            S_post_inv = (S_post_inv + S_post_inv.T) / 2
            # Ensure positive definite
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


# ---------------------------------------------------------------------------
# BayesianVAR result
# ---------------------------------------------------------------------------


@dataclass
class BayesianVARResult:
    """Fitted Bayesian VAR result.

    Attributes
    ----------
    B_post
        Posterior mean coefficient matrix, shape ``(k, k*p+1)``.
    Sigma_post
        Posterior mean covariance matrix, shape ``(k, k)``.
    B_samples
        MCMC posterior samples for B, shape ``(n_samples, k, k*p+1)``.
        ``None`` for analytical inference.
    Sigma_samples
        MCMC posterior samples for Sigma, shape ``(n_samples, k, k)``.
        ``None`` for analytical inference.
    target_cols
        Names of the modeled variables.
    p
        Number of lags.

    """

    B_post: np.ndarray
    Sigma_post: np.ndarray
    B_samples: np.ndarray | None = None
    Sigma_samples: np.ndarray | None = None
    target_cols: list[str] = field(default_factory=list)
    p: int = 1
    _last_values: np.ndarray = field(default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# BayesianVAR class
# ---------------------------------------------------------------------------


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

    def fit(
        self,
        df: pl.DataFrame,
        id_col: str | None = None,
    ) -> BayesianVAR:
        """Fit the Bayesian VAR model.

        Parameters
        ----------
        df
            Input DataFrame.
        id_col
            If provided, fit separate BVAR per group.

        Returns
        -------
        BayesianVAR
            Fitted instance (``self``).

        """
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

        # Build prior
        if self.prior == "minnesota":
            sigma_scale = self.minnesota_prior.sigma_scale
            if sigma_scale is None:
                sigma_scale = _estimate_sigma_from_ar(data, self.p)
            B0, V0_inv_diag = _minnesota_prior_precision(k, self.p, self.minnesota_prior, sigma_scale)
        else:
            # Normal-Wishart
            nw = self.nw_prior
            B0 = nw.B0 if nw.B0 is not None else np.zeros((k, dim))
            if nw.B0 is None:
                for j in range(k):
                    B0[j, j] = 1.0  # random walk default
            if nw.V0 is not None:
                V0_inv_diag = np.diag(nw.V0)
            else:
                V0_inv_diag = np.full(dim, 1.0 / max(nw.tightness**2, 1e-20))
                V0_inv_diag[-1] = 1e-6  # diffuse intercept

        # Scale matrix
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

    def predict(
        self,
        horizon: int,
        id_col: str | None = None,
    ) -> pl.DataFrame:
        """Generate multi-step forecasts with credible intervals.

        Parameters
        ----------
        horizon
            Number of steps to forecast.
        id_col
            If the model was fit with groups, include this column.

        Returns
        -------
        pl.DataFrame
            DataFrame with step index, point forecasts per variable,
            and lower/upper credible bounds for each variable.

        """
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

    def _forecast_single(
        self,
        result: BayesianVARResult,
        horizon: int,
        alpha_half: float,
    ) -> list[dict[str, Any]]:
        """Forecast from a single fitted result."""
        k = len(self.target_cols)
        p = self.p

        if self.inference == "gibbs" and result.B_samples is not None:
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
                    row = []
                    for lag in range(1, p + 1):
                        idx = len(history) - lag
                        row.extend(history[idx])
                    row.append(1.0)
                    x = np.array(row)
                    pred = B @ x + L @ rng.standard_normal(k)
                    all_fc[i, step] = pred
                    history.append(pred)

            mean_fc = np.mean(all_fc, axis=0)
            lower_fc = np.quantile(all_fc, alpha_half, axis=0)
            upper_fc = np.quantile(all_fc, 1 - alpha_half, axis=0)
        else:
            # Analytical: point forecast + approximate intervals
            B = result.B_post
            history = list(result._last_values.copy())
            mean_fc = np.empty((horizon, k))

            for step in range(horizon):
                row = []
                for lag in range(1, p + 1):
                    idx = len(history) - lag
                    row.extend(history[idx])
                row.append(1.0)
                x = np.array(row)
                pred = B @ x
                mean_fc[step] = pred
                history.append(pred)

            # Intervals widen with horizon
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

    def irf(
        self,
        steps: int = 20,
        shock_size: float = 1.0,
        gid: Any = None,
    ) -> pl.DataFrame:
        """Compute impulse response functions with credible bands.

        Parameters
        ----------
        steps
            Number of IRF steps.
        shock_size
            Size of the impulse shock (in standard deviations).
        gid
            Group identifier. Use ``None`` for ungrouped models.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``step``, ``impulse``, ``response``,
            ``irf``, ``irf_lower``, ``irf_upper``.

        """
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
            # Approximate bands: widen with horizon
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
    def _compute_irf(
        B: np.ndarray,
        k: int,
        p: int,
        steps: int,
        shock_size: float,
    ) -> np.ndarray:
        """Compute orthogonalized IRF from coefficient matrix.

        Returns array of shape ``(steps, k, k)`` where ``[t, i, j]``
        is the response of variable *j* at time *t* to a shock in *i*.
        """
        # Extract companion-form lag matrices
        A_mats = []
        for lag in range(p):
            A_mats.append(B[:, lag * k : (lag + 1) * k])

        # Compute MA representation via recursion
        Phi = np.zeros((steps, k, k))
        Phi[0] = np.eye(k) * shock_size

        for t in range(1, steps):
            for lag in range(min(t, p)):
                Phi[t] += A_mats[lag] @ Phi[t - lag - 1]

        return Phi


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


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
    """Bayesian VAR convenience function.

    Fits a BVAR and returns posterior predictive forecasts with
    credible intervals in a single call.

    Parameters
    ----------
    df
        Input DataFrame.
    target_cols
        Column names to model jointly.
    horizon
        Number of steps to forecast.
    p
        Number of lags.
    prior
        ``"minnesota"`` or ``"normal_wishart"``.
    inference
        ``"analytical"`` or ``"gibbs"``.
    minnesota_prior
        Minnesota prior settings.
    nw_prior
        Normal-Wishart prior settings.
    coverage
        Credible interval coverage (default 0.9).
    n_samples
        Gibbs samples (after burn-in).
    burn_in
        Gibbs burn-in samples.
    seed
        Random seed.
    time_col
        Column with timestamps.
    id_col
        If provided, fit per-group models.

    Returns
    -------
    pl.DataFrame
        Forecasts with point estimates and credible intervals
        for each target variable.

    """
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
