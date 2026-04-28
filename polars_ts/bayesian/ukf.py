"""Unscented Kalman Filter (UKF).

Propagates sigma points through user-defined nonlinear transition and
observation functions, then recovers the posterior mean and covariance
via a weighted combination.

Reference:
    Julier & Uhlmann (1997). *A New Extension of the Kalman Filter
    to Nonlinear Systems*.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from polars_ts.bayesian.kalman import KalmanResult


def _sigma_points(
    x: np.ndarray,
    P: np.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sigma points and weights for the unscented transform."""
    n = len(x)
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    if c <= 0:
        c = 1e-6  # fallback for very small alpha

    # Sigma points: 2n+1
    sigma = np.zeros((2 * n + 1, n))
    sigma[0] = x
    M = c * P
    # Ensure PD for Cholesky
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    if eigvals.min() < 1e-10:
        M += np.eye(n) * (1e-10 - eigvals.min())
    sqrt_P = np.linalg.cholesky(M)
    for i in range(n):
        sigma[i + 1] = x + sqrt_P[i]
        sigma[n + i + 1] = x - sqrt_P[i]

    # Weights
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)

    return sigma, Wm, Wc


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear state-space models.

    Parameters
    ----------
    f
        State transition function ``f(x) -> x_next``.
        Takes a 1D state array and returns the predicted state.
    h
        Observation function ``h(x) -> y``.
        Takes a 1D state array and returns the predicted observation.
    Q
        Process noise covariance ``(n, n)``.
    R
        Observation noise covariance ``(m, m)``.
    x0
        Initial state mean ``(n,)``.
    P0
        Initial state covariance ``(n, n)``.
    alpha
        Spread of sigma points around the mean. Default ``1e-3``.
    beta
        Prior knowledge of distribution (2.0 is optimal for Gaussian).
    kappa
        Secondary scaling parameter. Default ``0.0``.

    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def filter(self, y: np.ndarray) -> KalmanResult:
        """Run the UKF forward pass.

        Parameters
        ----------
        y
            Observations ``(T,)`` or ``(T, m)``. Use ``np.nan`` for missing.

        Returns
        -------
        KalmanResult

        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        T, m = y.shape
        n = len(self.x0)

        filtered_states = np.zeros((T, n))
        filtered_covs = np.zeros((T, n, n))
        predicted_states = np.zeros((T, n))
        predicted_covs = np.zeros((T, n, n))
        log_lik = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            # --- Predict ---
            sigma, Wm, Wc = _sigma_points(x, P, self.alpha, self.beta, self.kappa)
            sigma_pred = np.array([self.f(s) for s in sigma])

            x_pred = Wm @ sigma_pred
            P_pred = self.Q.copy()
            for i in range(len(Wm)):
                dx = sigma_pred[i] - x_pred
                P_pred += Wc[i] * np.outer(dx, dx)

            predicted_states[t] = x_pred
            predicted_covs[t] = P_pred

            yt = y[t]
            if np.any(np.isnan(yt)):
                x = x_pred
                P = P_pred
            else:
                # --- Update ---
                sigma2, Wm2, Wc2 = _sigma_points(x_pred, P_pred, self.alpha, self.beta, self.kappa)
                y_sigma = np.array([self.h(s) for s in sigma2])

                y_pred = Wm2 @ y_sigma
                S = self.R.copy()
                Pxy = np.zeros((n, m))
                for i in range(len(Wm2)):
                    dy = y_sigma[i] - y_pred
                    S += Wc2[i] * np.outer(dy, dy)
                    dx = sigma2[i] - x_pred
                    Pxy += Wc2[i] * np.outer(dx, dy)

                K = Pxy @ np.linalg.inv(S)
                innov = yt - y_pred
                x = x_pred + K @ innov
                P = P_pred - K @ S @ K.T
                # Ensure symmetry and positive-definiteness
                P = 0.5 * (P + P.T) + np.eye(n) * 1e-10

                sign, logdet = np.linalg.slogdet(S)
                log_lik += -0.5 * (m * np.log(2 * np.pi) + logdet + float(innov @ np.linalg.inv(S) @ innov))

            filtered_states[t] = x
            filtered_covs[t] = P

        return KalmanResult(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            log_likelihood=log_lik,
        )
