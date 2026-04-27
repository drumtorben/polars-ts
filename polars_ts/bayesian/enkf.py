"""Ensemble Kalman Filter (EnKF).

Monte Carlo ensemble propagation through nonlinear transition and
observation functions. Scalable to high-dimensional states where
maintaining a full covariance matrix is infeasible.

Reference:
    Evensen (2003). *The Ensemble Kalman Filter: theoretical formulation
    and practical implementation*.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from polars_ts.bayesian.kalman import KalmanResult


class EnsembleKalmanFilter:
    """Ensemble Kalman Filter for nonlinear state-space models.

    Parameters
    ----------
    f
        State transition function ``f(x) -> x_next``.
    h
        Observation function ``h(x) -> y``.
    Q
        Process noise covariance ``(n, n)``.
    R
        Observation noise covariance ``(m, m)``.
    x0
        Initial state mean ``(n,)``.
    P0
        Initial state covariance ``(n, n)``.
    n_ensemble
        Number of ensemble members. Default ``50``.
    seed
        Random seed for reproducibility.

    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        n_ensemble: int = 50,
        seed: int = 42,
    ) -> None:
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.n_ensemble = n_ensemble
        self.seed = seed

    def filter(self, y: np.ndarray) -> KalmanResult:
        """Run the EnKF forward pass.

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
        N = self.n_ensemble
        rng = np.random.default_rng(self.seed)

        # Initialize ensemble from prior
        L_P0 = np.linalg.cholesky(self.P0)
        ensemble = self.x0[None, :] + (rng.standard_normal((N, n)) @ L_P0.T)

        L_Q = np.linalg.cholesky(self.Q)
        L_R = np.linalg.cholesky(self.R)

        filtered_states = np.zeros((T, n))
        filtered_covs = np.zeros((T, n, n))
        predicted_states = np.zeros((T, n))
        predicted_covs = np.zeros((T, n, n))
        log_lik = 0.0

        for t in range(T):
            # --- Predict: propagate ensemble through f + process noise ---
            noise = rng.standard_normal((N, n)) @ L_Q.T
            ensemble_pred = np.array([self.f(ensemble[i]) for i in range(N)]) + noise

            x_pred = ensemble_pred.mean(axis=0)
            dx = ensemble_pred - x_pred
            P_pred = (dx.T @ dx) / (N - 1)

            predicted_states[t] = x_pred
            predicted_covs[t] = P_pred

            yt = y[t]
            if np.any(np.isnan(yt)):
                ensemble = ensemble_pred
            else:
                # --- Update: EnKF analysis step ---
                y_ensemble = np.array([self.h(ensemble_pred[i]) for i in range(N)])
                obs_noise = rng.standard_normal((N, m)) @ L_R.T
                y_perturbed = yt[None, :] + obs_noise

                y_mean = y_ensemble.mean(axis=0)
                dy = y_ensemble - y_mean

                # Cross-covariance and innovation covariance
                Pxy = (dx.T @ dy) / (N - 1)
                S = (dy.T @ dy) / (N - 1) + self.R

                # Kalman gain
                K = Pxy @ np.linalg.inv(S)

                # Update each ensemble member
                innovations = y_perturbed - y_ensemble
                ensemble = ensemble_pred + innovations @ K.T

                # Log-likelihood
                innov = yt - y_mean
                sign, logdet = np.linalg.slogdet(S)
                log_lik += -0.5 * (m * np.log(2 * np.pi) + logdet + float(innov @ np.linalg.inv(S) @ innov))

            x_filt = ensemble.mean(axis=0)
            dx_filt = ensemble - x_filt
            P_filt = (dx_filt.T @ dx_filt) / (N - 1)

            filtered_states[t] = x_filt
            filtered_covs[t] = P_filt

        return KalmanResult(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            log_likelihood=log_lik,
        )
