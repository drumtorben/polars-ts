"""Kalman Filter and Rauch-Tung-Striebel (RTS) smoother.

Linear Gaussian state-space model:

    x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)   (state transition)
    y_t = H @ x_t     + v_t,  v_t ~ N(0, R)   (observation)

The Kalman filter computes filtered state estimates (forward pass),
and the RTS smoother refines them (backward pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class KalmanResult:
    """Container for Kalman filter / smoother output.

    Attributes
    ----------
    filtered_states
        Array of shape ``(T, n)`` — filtered state means.
    filtered_covs
        Array of shape ``(T, n, n)`` — filtered state covariances.
    predicted_states
        Array of shape ``(T, n)`` — one-step-ahead predicted state means.
    predicted_covs
        Array of shape ``(T, n, n)`` — one-step-ahead predicted covariances.
    smoothed_states
        Array of shape ``(T, n)`` — smoothed state means (after RTS pass).
        ``None`` if smoothing was not requested.
    smoothed_covs
        Array of shape ``(T, n, n)`` — smoothed state covariances.
        ``None`` if smoothing was not requested.
    log_likelihood
        Total log-likelihood of the observations.

    """

    filtered_states: np.ndarray
    filtered_covs: np.ndarray
    predicted_states: np.ndarray
    predicted_covs: np.ndarray
    smoothed_states: np.ndarray | None = None
    smoothed_covs: np.ndarray | None = None
    log_likelihood: float = 0.0


class KalmanFilter:
    """Kalman Filter with optional RTS smoother.

    Parameters
    ----------
    F
        State transition matrix of shape ``(n, n)``.
    H
        Observation matrix of shape ``(m, n)`` where ``m`` is the
        observation dimension and ``n`` is the state dimension.
    Q
        Process noise covariance of shape ``(n, n)``.
    R
        Observation noise covariance of shape ``(m, m)``.
    x0
        Initial state mean of shape ``(n,)``. Defaults to zeros.
    P0
        Initial state covariance of shape ``(n, n)``.
        Defaults to ``1e6 * I`` (diffuse prior).

    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        n = self.F.shape[0]
        self.x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
        self.P0 = np.eye(n) * 1e6 if P0 is None else np.asarray(P0, dtype=np.float64)

    def filter(self, y: np.ndarray) -> KalmanResult:
        """Run the Kalman filter forward pass.

        Parameters
        ----------
        y
            Observations of shape ``(T,)`` or ``(T, m)``.
            Use ``np.nan`` for missing observations.

        Returns
        -------
        KalmanResult
            Filtered states, covariances, and log-likelihood.

        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        T, m = y.shape
        n = self.F.shape[0]

        filtered_states = np.zeros((T, n))
        filtered_covs = np.zeros((T, n, n))
        predicted_states = np.zeros((T, n))
        predicted_covs = np.zeros((T, n, n))
        log_lik = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            predicted_states[t] = x_pred
            predicted_covs[t] = P_pred

            # Check for missing observation
            yt = y[t]
            if np.any(np.isnan(yt)):
                # No update — predicted = filtered
                x = x_pred
                P = P_pred
            else:
                # Innovation
                innov = yt - self.H @ x_pred
                S = self.H @ P_pred @ self.H.T + self.R

                # Kalman gain
                S_inv = np.linalg.inv(S)
                K = P_pred @ self.H.T @ S_inv

                # Update
                x = x_pred + K @ innov
                P = (np.eye(n) - K @ self.H) @ P_pred

                # Log-likelihood contribution
                sign, logdet = np.linalg.slogdet(S)
                log_lik += -0.5 * (m * np.log(2 * np.pi) + logdet + float(innov.T @ S_inv @ innov))

            filtered_states[t] = x
            filtered_covs[t] = P

        return KalmanResult(
            filtered_states=filtered_states,
            filtered_covs=filtered_covs,
            predicted_states=predicted_states,
            predicted_covs=predicted_covs,
            log_likelihood=log_lik,
        )

    def smooth(self, y: np.ndarray) -> KalmanResult:
        """Run the Kalman filter + RTS smoother.

        Parameters
        ----------
        y
            Observations of shape ``(T,)`` or ``(T, m)``.

        Returns
        -------
        KalmanResult
            Filtered and smoothed states, covariances, and log-likelihood.

        """
        result = self.filter(y)
        T = result.filtered_states.shape[0]
        n = self.F.shape[0]

        smoothed_states = np.zeros((T, n))
        smoothed_covs = np.zeros((T, n, n))

        # Initialise from the last filtered state
        smoothed_states[-1] = result.filtered_states[-1]
        smoothed_covs[-1] = result.filtered_covs[-1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            P_pred = result.predicted_covs[t + 1]
            P_filt = result.filtered_covs[t]

            # Smoother gain
            L = P_filt @ self.F.T @ np.linalg.inv(P_pred)

            smoothed_states[t] = result.filtered_states[t] + L @ (
                smoothed_states[t + 1] - result.predicted_states[t + 1]
            )
            smoothed_covs[t] = P_filt + L @ (smoothed_covs[t + 1] - P_pred) @ L.T

        result.smoothed_states = smoothed_states
        result.smoothed_covs = smoothed_covs
        return result


def kalman_filter(
    df: pl.DataFrame,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
    smooth: bool = True,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, KalmanResult]:
    """Apply Kalman filter (and optional RTS smoother) to each time series.

    Convenience function that wraps :class:`KalmanFilter` for panel data.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    F
        State transition matrix ``(n, n)``.
    H
        Observation matrix ``(m, n)``.
    Q
        Process noise covariance ``(n, n)``.
    R
        Observation noise covariance ``(m, m)``.
    x0
        Initial state mean ``(n,)``. Defaults to zeros.
    P0
        Initial state covariance ``(n, n)``. Defaults to diffuse prior.
    smooth
        Whether to run the RTS smoother after filtering.
    id_col
        Column identifying each time series.
    target_col
        Column with the observed values.

    Returns
    -------
    dict[str, KalmanResult]
        Mapping from series ID to the filter/smoother result.

    """
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
    results: dict[str, KalmanResult] = {}

    for sid in df[id_col].unique(maintain_order=True).to_list():
        y = df.filter(pl.col(id_col) == sid)[target_col].to_numpy()
        if smooth:
            results[str(sid)] = kf.smooth(y)
        else:
            results[str(sid)] = kf.filter(y)

    return results
