"""GARCH volatility modelling for time series. Closes #51."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class GARCHResult:
    """Fitted GARCH model result."""

    omega: float
    alpha: list[float]
    beta: list[float]
    p: int
    q: int
    conditional_variance: list[float] = field(default_factory=list)
    log_likelihood: float = 0.0


def garch_fit(
    df: pl.DataFrame,
    p: int = 1,
    q: int = 1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    max_iter: int = 200,
) -> dict[Any, GARCHResult]:
    """Fit a GARCH(p,q) model to each time series.

    Parameters
    ----------
    df
        Input DataFrame (typically returns or residuals).
    p
        Number of lagged squared residuals (ARCH terms).
    q
        Number of lagged conditional variances (GARCH terms).
    target_col
        Column with the values to model.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    max_iter
        Maximum optimization iterations.

    Returns
    -------
    dict
        Mapping from series ID to :class:`GARCHResult`.

    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if q < 0:
        raise ValueError("q must be >= 0")

    from scipy.optimize import minimize

    sorted_df = df.sort(id_col, time_col)
    results: dict[Any, GARCHResult] = {}

    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        data = np.array(group_df[target_col].to_list(), dtype=np.float64)
        n = len(data)

        if n < p + q + 1:
            raise ValueError(f"Series {gid!r} too short for GARCH({p},{q})")

        # Demean
        eps = data - data.mean()
        var_eps = float(np.var(eps))

        def neg_log_lik(params: np.ndarray, *, _n: int = n, _eps: np.ndarray = eps, _var_eps: float = var_eps) -> float:
            omega = params[0]
            a = params[1 : 1 + p]
            b = params[1 + p : 1 + p + q]

            # Enforce positivity
            if omega <= 0 or np.any(a < 0) or np.any(b < 0):
                return 1e10
            if np.sum(a) + np.sum(b) >= 1:
                return 1e10

            sigma2 = np.full(_n, _var_eps)
            for t in range(max(p, q), _n):
                sigma2[t] = omega
                for i in range(p):
                    sigma2[t] += a[i] * _eps[t - 1 - i] ** 2
                for j in range(q):
                    sigma2[t] += b[j] * sigma2[t - 1 - j]
                if sigma2[t] <= 0:
                    return 1e10

            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + _eps**2 / sigma2)
            return -ll

        # Initialize
        x0 = np.zeros(1 + p + q)
        x0[0] = var_eps * 0.1  # omega
        x0[1 : 1 + p] = 0.1  # alpha
        if q > 0:
            x0[1 + p : 1 + p + q] = 0.8 / q  # beta

        res = minimize(neg_log_lik, x0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-8})

        omega = float(res.x[0])
        alpha_vals = res.x[1 : 1 + p].tolist()
        beta_vals = res.x[1 + p : 1 + p + q].tolist()

        # Compute conditional variance with fitted params
        sigma2 = np.full(n, var_eps)
        for t in range(max(p, q), n):
            sigma2[t] = omega
            for i in range(p):
                sigma2[t] += alpha_vals[i] * eps[t - 1 - i] ** 2
            for j in range(q):
                sigma2[t] += beta_vals[j] * sigma2[t - 1 - j]

        results[gid] = GARCHResult(
            omega=omega,
            alpha=alpha_vals,
            beta=beta_vals,
            p=p,
            q=q,
            conditional_variance=sigma2.tolist(),
            log_likelihood=-float(res.fun),
        )

    return results


def garch_forecast(
    model: GARCHResult,
    horizon: int,
) -> list[float]:
    """Forecast conditional variance for *horizon* steps ahead.

    Parameters
    ----------
    model
        A fitted :class:`GARCHResult`.
    horizon
        Number of steps to forecast.

    Returns
    -------
    list[float]
        Forecast conditional variances.

    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    sigma2 = list(model.conditional_variance)
    # For forecasting, assume future squared residuals equal conditional variance
    for _step in range(horizon):
        val = model.omega
        for i, a in enumerate(model.alpha):
            idx = len(sigma2) - 1 - i
            val += a * sigma2[idx] if idx >= 0 else 0.0
        for j, b in enumerate(model.beta):
            idx = len(sigma2) - 1 - j
            val += b * sigma2[idx] if idx >= 0 else 0.0
        sigma2.append(val)

    return sigma2[-horizon:]
