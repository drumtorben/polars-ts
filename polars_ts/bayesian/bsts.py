"""Bayesian Structural Time Series (BSTS).

A flexible state-space framework that decomposes a time series into
structural components (level, trend, seasonality, regression) and
estimates them via the Kalman filter/smoother.

The state vector is assembled by stacking component sub-states:

    x_t = [level_t, trend_t, season_t(1), ..., season_t(s-1), beta ...]

References:
    Scott & Varian (2014). *Predicting the Present with Bayesian
    Structural Time Series*.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_ts.bayesian.kalman import KalmanFilter, KalmanResult


@dataclass
class BSTSResult:
    """Container for BSTS fit / forecast output.

    Attributes
    ----------
    kalman_result
        Full Kalman filter/smoother result.
    level
        Smoothed level component of shape ``(T,)``.
    trend
        Smoothed trend component of shape ``(T,)`` or ``None``.
    seasonal
        Smoothed seasonal component of shape ``(T,)`` or ``None``.
    regression
        Smoothed regression component of shape ``(T,)`` or ``None``.
    forecast
        Point forecast of shape ``(h,)`` or ``None``.
    forecast_var
        Forecast variance of shape ``(h,)`` or ``None``.

    """

    kalman_result: KalmanResult
    level: np.ndarray
    trend: np.ndarray | None = None
    seasonal: np.ndarray | None = None
    regression: np.ndarray | None = None
    forecast: np.ndarray | None = None
    forecast_var: np.ndarray | None = None


def _build_local_level(sigma_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q) blocks for a local-level component."""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[sigma_level**2]])
    return F, H, Q


def _build_local_linear_trend(sigma_level: float, sigma_trend: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q) blocks for a local linear trend."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([sigma_level**2, sigma_trend**2])
    return F, H, Q


def _build_seasonal(n_seasons: int, sigma_seasonal: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, H, Q) blocks for a dummy seasonal component.

    State dimension is ``n_seasons - 1``. The constraint is that
    seasonal effects sum to zero over a full cycle.
    """
    s = n_seasons - 1
    F = np.zeros((s, s))
    F[0, :] = -1.0
    if s > 1:
        F[1:, :-1] = np.eye(s - 1)
    H = np.zeros((1, s))
    H[0, 0] = 1.0
    Q = np.zeros((s, s))
    Q[0, 0] = sigma_seasonal**2
    return F, H, Q


def _block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Build a block-diagonal matrix from sub-blocks."""
    sizes = [b.shape[0] for b in blocks]
    total = sum(sizes)
    result = np.zeros((total, total))
    offset = 0
    for b in blocks:
        n = b.shape[0]
        result[offset : offset + n, offset : offset + n] = b
        offset += n
    return result


class BSTS:
    """Bayesian Structural Time Series model.

    Assembles a state-space model from structural components and
    estimates states via the Kalman filter/smoother.

    Parameters
    ----------
    trend
        Trend type: ``"level"`` (random walk) or ``"local_linear"``
        (random walk + drift).
    seasonal
        Number of seasons. ``None`` to disable seasonality.
    sigma_obs
        Observation noise standard deviation.
    sigma_level
        Level component noise standard deviation.
    sigma_trend
        Trend component noise standard deviation (only for
        ``trend="local_linear"``).
    sigma_seasonal
        Seasonal component noise standard deviation.

    """

    def __init__(
        self,
        trend: str = "local_linear",
        seasonal: int | None = None,
        sigma_obs: float = 1.0,
        sigma_level: float = 0.1,
        sigma_trend: float = 0.01,
        sigma_seasonal: float = 0.01,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.sigma_obs = sigma_obs
        self.sigma_level = sigma_level
        self.sigma_trend = sigma_trend
        self.sigma_seasonal = sigma_seasonal

    def _build_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble full system matrices from components."""
        F_blocks: list[np.ndarray] = []
        H_blocks: list[np.ndarray] = []
        Q_blocks: list[np.ndarray] = []

        if self.trend == "level":
            F, H, Q = _build_local_level(self.sigma_level)
        elif self.trend == "local_linear":
            F, H, Q = _build_local_linear_trend(self.sigma_level, self.sigma_trend)
        else:
            raise ValueError(f"Unknown trend type {self.trend!r}. Supported: 'level', 'local_linear'")

        F_blocks.append(F)
        H_blocks.append(H)
        Q_blocks.append(Q)

        if self.seasonal is not None:
            Fs, Hs, Qs = _build_seasonal(self.seasonal, self.sigma_seasonal)
            F_blocks.append(Fs)
            H_blocks.append(Hs)
            Q_blocks.append(Qs)

        F_full = _block_diag(*F_blocks)
        Q_full = _block_diag(*Q_blocks)
        H_full = np.hstack(H_blocks)
        R = np.array([[self.sigma_obs**2]])

        return F_full, H_full, Q_full, R

    def fit(self, y: np.ndarray) -> BSTSResult:
        """Fit the BSTS model to observed data.

        Parameters
        ----------
        y
            Observations of shape ``(T,)``. Use ``np.nan`` for missing.

        Returns
        -------
        BSTSResult
            Decomposed components and Kalman filter/smoother result.

        """
        F, H, Q, R = self._build_system()
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
        kr = kf.smooth(y)

        assert kr.smoothed_states is not None
        states = kr.smoothed_states

        # Extract components
        idx = 0
        if self.trend == "level":
            level = states[:, 0]
            trend_component = None
            idx = 1
        else:
            level = states[:, 0]
            trend_component = states[:, 1]
            idx = 2

        seasonal_component = None
        if self.seasonal is not None:
            seasonal_component = states[:, idx]

        return BSTSResult(
            kalman_result=kr,
            level=level,
            trend=trend_component,
            seasonal=seasonal_component,
        )

    def forecast(self, y: np.ndarray, h: int) -> BSTSResult:
        """Fit the model and produce h-step-ahead forecasts.

        Parameters
        ----------
        y
            Observations of shape ``(T,)``.
        h
            Number of steps to forecast.

        Returns
        -------
        BSTSResult
            Components plus ``forecast`` and ``forecast_var`` arrays.

        """
        result = self.fit(y)
        kr = result.kalman_result

        F, H, Q, R = self._build_system()

        # Forward-propagate from last filtered state
        x = kr.filtered_states[-1]
        P = kr.filtered_covs[-1]

        forecasts = np.zeros(h)
        variances = np.zeros(h)

        for t in range(h):
            x = F @ x
            P = F @ P @ F.T + Q
            y_pred = H @ x
            y_var = H @ P @ H.T + R
            forecasts[t] = float(y_pred[0])
            variances[t] = float(y_var[0, 0])

        result.forecast = forecasts
        result.forecast_var = variances
        return result


def bsts_fit(
    df: pl.DataFrame,
    trend: str = "local_linear",
    seasonal: int | None = None,
    sigma_obs: float = 1.0,
    sigma_level: float = 0.1,
    sigma_trend: float = 0.01,
    sigma_seasonal: float = 0.01,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, BSTSResult]:
    """Fit a BSTS model to each time series in a panel.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    trend
        ``"level"`` or ``"local_linear"``.
    seasonal
        Number of seasons (e.g. 12 for monthly, 24 for hourly).
        ``None`` to disable.
    sigma_obs, sigma_level, sigma_trend, sigma_seasonal
        Noise standard deviations for each component.
    id_col
        Column identifying each time series.
    target_col
        Column with the observed values.

    Returns
    -------
    dict[str, BSTSResult]
        Mapping from series ID to the fit result.

    """
    model = BSTS(
        trend=trend,
        seasonal=seasonal,
        sigma_obs=sigma_obs,
        sigma_level=sigma_level,
        sigma_trend=sigma_trend,
        sigma_seasonal=sigma_seasonal,
    )
    results: dict[str, BSTSResult] = {}
    for sid in df[id_col].unique(maintain_order=True).to_list():
        y = df.filter(pl.col(id_col) == sid)[target_col].to_numpy()
        results[str(sid)] = model.fit(y)
    return results


def bsts_forecast(
    df: pl.DataFrame,
    h: int,
    trend: str = "local_linear",
    seasonal: int | None = None,
    sigma_obs: float = 1.0,
    sigma_level: float = 0.1,
    sigma_trend: float = 0.01,
    sigma_seasonal: float = 0.01,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, BSTSResult]:
    """Fit a BSTS model and produce h-step-ahead forecasts.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    h
        Forecast horizon.
    trend, seasonal, sigma_obs, sigma_level, sigma_trend, sigma_seasonal
        Model configuration (see :class:`BSTS`).
    id_col
        Column identifying each time series.
    target_col
        Column with the observed values.

    Returns
    -------
    dict[str, BSTSResult]
        Mapping from series ID to the fit+forecast result.

    """
    model = BSTS(
        trend=trend,
        seasonal=seasonal,
        sigma_obs=sigma_obs,
        sigma_level=sigma_level,
        sigma_trend=sigma_trend,
        sigma_seasonal=sigma_seasonal,
    )
    results: dict[str, BSTSResult] = {}
    for sid in df[id_col].unique(maintain_order=True).to_list():
        y = df.filter(pl.col(id_col) == sid)[target_col].to_numpy()
        results[str(sid)] = model.forecast(y, h=h)
    return results
