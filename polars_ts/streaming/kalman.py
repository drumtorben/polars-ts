"""Online Kalman filter with single-observation updates."""

from __future__ import annotations

import numpy as np


class StreamingKalmanFilter:
    """Kalman filter supporting incremental single-observation updates.

    Extends the batch KalmanFilter with an ``update()`` method for
    processing observations one at a time in a streaming context.

    Parameters
    ----------
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
        self._x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
        self._P0 = np.eye(n) * 1e6 if P0 is None else np.asarray(P0, dtype=np.float64)

        self.is_fitted_ = False
        self.state_mean: np.ndarray = self._x0.copy()
        self.state_cov: np.ndarray = self._P0.copy()
        self.log_likelihood_: float = 0.0

    def fit(self, y: np.ndarray) -> StreamingKalmanFilter:
        """Initialize filter by processing a batch of observations.

        Parameters
        ----------
        y
            Observations array of shape ``(T,)`` or ``(T, m)``.

        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = y.shape[1]
        n = self.F.shape[0]

        x = self._x0.copy()
        P = self._P0.copy()
        log_lik = 0.0

        for t in range(y.shape[0]):
            x, P, ll_contrib = self._step(x, P, y[t], n, m)
            log_lik += ll_contrib

        self.state_mean = x
        self.state_cov = P
        self.log_likelihood_ = log_lik
        self.is_fitted_ = True
        return self

    def update(self, observation: float | np.ndarray) -> StreamingKalmanFilter:
        """Process a single new observation, updating state in-place.

        Parameters
        ----------
        observation
            Scalar or array of shape ``(m,)``.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before update()")

        obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
        n = self.F.shape[0]
        m = obs.shape[0]

        x, P, ll_contrib = self._step(self.state_mean, self.state_cov, obs, n, m)
        self.state_mean = x
        self.state_cov = P
        self.log_likelihood_ += ll_contrib
        return self

    def predict(self, h: int) -> np.ndarray:
        """Predict h steps ahead from current state.

        Returns
        -------
        predictions
            Array of shape ``(h,)`` — predicted observation means.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")

        predictions = np.zeros(h)
        x = self.state_mean.copy()
        for step in range(h):
            x = self.F @ x
            predictions[step] = (self.H @ x)[0]
        return predictions

    def _step(
        self,
        x: np.ndarray,
        P: np.ndarray,
        yt: np.ndarray,
        n: int,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Single predict-update cycle."""
        # Predict
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q

        # Handle missing
        if np.any(np.isnan(yt)):
            return x_pred, P_pred, 0.0

        # Innovation
        innov = yt - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        S_inv = np.linalg.inv(S)
        K = P_pred @ self.H.T @ S_inv

        # Update
        x_new = x_pred + K @ innov
        P_new = (np.eye(n) - K @ self.H) @ P_pred

        # Log-likelihood contribution
        sign, logdet = np.linalg.slogdet(S)
        ll = -0.5 * (m * np.log(2 * np.pi) + logdet + float(innov.T @ S_inv @ innov))

        return x_new, P_new, ll
