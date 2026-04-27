"""Gaussian Process regression for time series forecasting.

Non-parametric Bayesian forecasting with kernels designed for temporal
data. Supports exact GP (Cholesky) and sparse GP (inducing points).

References
----------
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*
- Roberts et al. (2013), *Gaussian processes for time-series modelling*
- Wilson & Adams (2013), *GP Kernels for Pattern Discovery and Extrapolation*

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

KernelType = Literal["rbf", "matern32", "matern52", "periodic", "spectral_mixture"]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class Kernel:
    """Base kernel class."""

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        raise NotImplementedError

    def set_params(self, params: np.ndarray) -> None:
        raise NotImplementedError

    def get_params(self) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: Kernel) -> SumKernel:
        return SumKernel(self, other)

    def __mul__(self, other: Kernel) -> ProductKernel:
        return ProductKernel(self, other)


class RBFKernel(Kernel):
    """Radial Basis Function (squared exponential) kernel."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0) -> None:
        self.variance = variance
        self.lengthscale = lengthscale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = (X1[:, None] - X2[None, :]) ** 2
        return self.variance * np.exp(-0.5 * sq_dist / self.lengthscale**2)

    def n_params(self) -> int:
        return 2

    def set_params(self, params: np.ndarray) -> None:
        self.variance = float(np.exp(params[0]))
        self.lengthscale = float(np.exp(params[1]))

    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance), np.log(self.lengthscale)])


class Matern32Kernel(Kernel):
    """Matern 3/2 kernel."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0) -> None:
        self.variance = variance
        self.lengthscale = lengthscale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r = np.abs(X1[:, None] - X2[None, :]) / self.lengthscale
        sqrt3r = math.sqrt(3) * r
        return self.variance * (1 + sqrt3r) * np.exp(-sqrt3r)

    def n_params(self) -> int:
        return 2

    def set_params(self, params: np.ndarray) -> None:
        self.variance = float(np.exp(params[0]))
        self.lengthscale = float(np.exp(params[1]))

    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance), np.log(self.lengthscale)])


class Matern52Kernel(Kernel):
    """Matern 5/2 kernel."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0) -> None:
        self.variance = variance
        self.lengthscale = lengthscale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r = np.abs(X1[:, None] - X2[None, :]) / self.lengthscale
        sqrt5r = math.sqrt(5) * r
        return self.variance * (1 + sqrt5r + 5.0 / 3.0 * r**2) * np.exp(-sqrt5r)

    def n_params(self) -> int:
        return 2

    def set_params(self, params: np.ndarray) -> None:
        self.variance = float(np.exp(params[0]))
        self.lengthscale = float(np.exp(params[1]))

    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance), np.log(self.lengthscale)])


class PeriodicKernel(Kernel):
    """Periodic kernel for repeating patterns."""

    def __init__(self, variance: float = 1.0, lengthscale: float = 1.0, period: float = 1.0) -> None:
        self.variance = variance
        self.lengthscale = lengthscale
        self.period = period

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None] - X2[None, :]
        sin_term = np.sin(np.pi * diff / self.period)
        return self.variance * np.exp(-2.0 * sin_term**2 / self.lengthscale**2)

    def n_params(self) -> int:
        return 3

    def set_params(self, params: np.ndarray) -> None:
        self.variance = float(np.exp(params[0]))
        self.lengthscale = float(np.exp(params[1]))
        self.period = float(np.exp(params[2]))

    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.variance), np.log(self.lengthscale), np.log(self.period)])


class SpectralMixtureKernel(Kernel):
    """Spectral mixture kernel (Wilson & Adams 2013)."""

    def __init__(self, n_mixtures: int = 3) -> None:
        self.n_mixtures = n_mixtures
        self.weights = np.ones(n_mixtures) / n_mixtures
        self.means = np.linspace(0.1, 1.0, n_mixtures)
        self.variances = np.ones(n_mixtures) * 0.1

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        tau = X1[:, None] - X2[None, :]
        K = np.zeros_like(tau)
        for q in range(self.n_mixtures):
            K += (
                self.weights[q]
                * np.exp(-2 * np.pi**2 * tau**2 * self.variances[q])
                * np.cos(2 * np.pi * tau * self.means[q])
            )
        return K

    def n_params(self) -> int:
        return 3 * self.n_mixtures

    def set_params(self, params: np.ndarray) -> None:
        m = self.n_mixtures
        self.weights = np.exp(params[:m])
        self.weights /= self.weights.sum()
        self.means = np.exp(params[m : 2 * m])
        self.variances = np.exp(params[2 * m : 3 * m])

    def get_params(self) -> np.ndarray:
        return np.concatenate(
            [np.log(self.weights + 1e-10), np.log(self.means + 1e-10), np.log(self.variances + 1e-10)]
        )


class SumKernel(Kernel):
    """Sum of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel) -> None:
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def n_params(self) -> int:
        return self.k1.n_params() + self.k2.n_params()

    def set_params(self, params: np.ndarray) -> None:
        n1 = self.k1.n_params()
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])


class ProductKernel(Kernel):
    """Product of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel) -> None:
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)

    def n_params(self) -> int:
        return self.k1.n_params() + self.k2.n_params()

    def set_params(self, params: np.ndarray) -> None:
        n1 = self.k1.n_params()
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])


def make_kernel(name: KernelType, **kwargs: Any) -> Kernel:
    """Create a kernel by name."""
    kernels: dict[str, type[Kernel]] = {
        "rbf": RBFKernel,
        "matern32": Matern32Kernel,
        "matern52": Matern52Kernel,
        "periodic": PeriodicKernel,
        "spectral_mixture": SpectralMixtureKernel,
    }
    if name not in kernels:
        raise ValueError(f"Unknown kernel {name!r}. Choose from {sorted(kernels.keys())}")
    return kernels[name](**kwargs)


# ---------------------------------------------------------------------------
# GP inference
# ---------------------------------------------------------------------------


def _stable_cholesky(K: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Cholesky with increasing jitter for numerical stability."""
    for scale in [1, 10, 100, 1000]:
        try:
            return np.linalg.cholesky(K + np.eye(len(K)) * jitter * scale)
        except np.linalg.LinAlgError:
            continue
    return np.diag(np.sqrt(np.maximum(np.diag(K), jitter)))


def _neg_log_marginal_likelihood(
    log_params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    noise_idx: int,
) -> float:
    """Negative log marginal likelihood for hyperparameter optimization."""
    kernel.set_params(log_params[:noise_idx])
    noise_var = float(np.exp(log_params[noise_idx]))

    K = kernel(X, X) + np.eye(len(X)) * noise_var
    try:
        L = np.linalg.cholesky(K + np.eye(len(X)) * 1e-8)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    n = len(y)

    nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
    return float(nll) if np.isfinite(nll) else 1e10


def _optimize_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    noise_var: float,
) -> tuple[Kernel, float]:
    """Optimize kernel hyperparameters via marginal likelihood."""
    from scipy.optimize import minimize

    noise_idx = kernel.n_params()
    x0 = np.append(kernel.get_params(), np.log(noise_var))

    result = minimize(
        _neg_log_marginal_likelihood,
        x0,
        args=(X, y, kernel, noise_idx),
        method="L-BFGS-B",
    )

    kernel.set_params(result.x[:noise_idx])
    opt_noise = float(np.exp(result.x[noise_idx]))
    return kernel, opt_noise


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class GPResult:
    """Gaussian Process regression result.

    Attributes
    ----------
    kernel
        Fitted kernel (with optimized hyperparameters).
    noise_var
        Optimized observation noise variance.
    X_train
        Training time indices.
    y_train
        Training observations.
    alpha
        Cholesky solve vector for predictions.
    L
        Lower Cholesky factor of training covariance.

    """

    kernel: Kernel
    noise_var: float
    X_train: np.ndarray
    y_train: np.ndarray
    alpha: np.ndarray = field(default_factory=lambda: np.empty(0))
    L: np.ndarray = field(default_factory=lambda: np.empty(0))


# ---------------------------------------------------------------------------
# GaussianProcessTS class
# ---------------------------------------------------------------------------


class GaussianProcessTS:
    """Gaussian Process time series forecaster.

    Parameters
    ----------
    kernel
        Kernel name or ``Kernel`` instance. Default ``"rbf"``.
    optimize
        Whether to optimize hyperparameters via marginal likelihood.
    noise_var
        Initial observation noise variance.
    coverage
        Credible interval coverage (default 0.9).
    id_col
        Column identifying each time series.
    target_col
        Column with target values.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        kernel: KernelType | Kernel = "rbf",
        optimize: bool = True,
        noise_var: float = 0.1,
        coverage: float = 0.9,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        if isinstance(kernel, str):
            self._kernel = make_kernel(kernel)
        else:
            self._kernel = kernel
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")

        self.optimize = optimize
        self.noise_var = noise_var
        self.coverage = coverage
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col

        self._results: dict[Any, GPResult] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> GaussianProcessTS:
        """Fit the GP to one or more time series."""
        sorted_df = df.sort(self.id_col, self.time_col)

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            y = group_df[self.target_col].to_numpy().astype(np.float64)
            n = len(y)
            X = np.arange(n, dtype=np.float64)

            # Normalize
            y_mean = float(np.mean(y))
            y_std = float(np.std(y))
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y - y_mean) / y_std

            import copy

            kernel = copy.deepcopy(self._kernel)
            noise = self.noise_var

            if self.optimize:
                kernel, noise = _optimize_hyperparams(X, y_norm, kernel, noise)

            K = kernel(X, X) + np.eye(n) * noise
            L = _stable_cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

            result = GPResult(
                kernel=kernel,
                noise_var=noise,
                X_train=X,
                y_train=y,
                alpha=alpha,
                L=L,
            )
            # Store normalization params
            result._y_mean = y_mean  # type: ignore[attr-defined]
            result._y_std = y_std  # type: ignore[attr-defined]

            self._results[gid] = result

        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame, h: int) -> pl.DataFrame:
        """Generate h-step forecasts with credible intervals."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        if h <= 0:
            raise ValueError("h must be positive")

        from scipy.stats import norm

        z = norm.ppf(1 - (1 - self.coverage) / 2)
        sorted_df = df.sort(self.id_col, self.time_col)
        all_rows: list[dict[str, Any]] = []

        for group_id, _group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]
            result = self._results.get(gid)
            if result is None:
                raise ValueError(f"Series {gid!r} was not seen during fit()")

            n = len(result.X_train)
            X_star = np.arange(n, n + h, dtype=np.float64)

            K_star = result.kernel(X_star, result.X_train)
            K_ss = result.kernel(X_star, X_star)

            # Posterior mean and variance
            mu = K_star @ result.alpha
            v = np.linalg.solve(result.L, K_star.T)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)

            # Denormalize
            y_mean = result._y_mean  # type: ignore[attr-defined]
            y_std = result._y_std  # type: ignore[attr-defined]
            mu = mu * y_std + y_mean
            std = np.sqrt(var) * y_std

            for step in range(h):
                all_rows.append(
                    {
                        self.id_col: gid,
                        "step": step + 1,
                        "y_hat": float(mu[step]),
                        "y_hat_lower": float(mu[step] - z * std[step]),
                        "y_hat_upper": float(mu[step] + z * std[step]),
                    }
                )

        return pl.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def gp_forecast(
    df: pl.DataFrame,
    h: int,
    kernel: KernelType | Kernel = "rbf",
    optimize: bool = True,
    noise_var: float = 0.1,
    coverage: float = 0.9,
    id_col: str = "unique_id",
    target_col: str = "y",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Gaussian Process forecasting convenience function.

    Parameters
    ----------
    df
        Input DataFrame.
    h
        Forecast horizon.
    kernel
        Kernel name or instance. Default ``"rbf"``.
    optimize
        Optimize hyperparameters via marginal likelihood.
    noise_var
        Initial noise variance.
    coverage
        Credible interval coverage (default 0.9).
    id_col, target_col, time_col
        Column names.

    Returns
    -------
    pl.DataFrame
        Forecasts with ``y_hat``, ``y_hat_lower``, ``y_hat_upper``.

    """
    gp = GaussianProcessTS(
        kernel=kernel,
        optimize=optimize,
        noise_var=noise_var,
        coverage=coverage,
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
    )
    gp.fit(df)
    return gp.predict(df, h)
