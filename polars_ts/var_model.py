"""Vector Autoregression (VAR) for multivariate time series. Closes #50."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class VARResult:
    """Fitted VAR model."""

    coefficients: np.ndarray  # shape (k, k*p + 1) — includes intercept
    target_cols: list[str]
    p: int
    residuals: np.ndarray = field(default_factory=lambda: np.empty(0))
    _last_values: np.ndarray = field(default_factory=lambda: np.empty(0))


def var_fit(
    df: pl.DataFrame,
    target_cols: list[str],
    p: int = 1,
    time_col: str = "ds",
) -> VARResult:
    """Fit a VAR(p) model via OLS.

    Parameters
    ----------
    df
        Input DataFrame with all target columns and a time column.
        Should represent a single multivariate series (no id_col grouping).
    target_cols
        List of column names to model jointly.
    p
        Number of lags.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    VARResult
        Fitted model containing coefficient matrix and residuals.

    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if len(target_cols) < 2:
        raise ValueError("VAR requires at least 2 target columns")

    sorted_df = df.sort(time_col)
    data = sorted_df.select(target_cols).to_numpy().astype(np.float64)
    n = len(data)

    if n <= p:
        raise ValueError(f"Need more than {p} observations for VAR({p})")

    # Build design matrix: each row has [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1]
    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []

    for t in range(p, n):
        row = []
        for lag in range(1, p + 1):
            row.extend(data[t - lag])
        row.append(1.0)  # intercept
        X_rows.append(np.array(row))
        Y_rows.append(data[t])

    X = np.array(X_rows)
    Y = np.array(Y_rows)

    # OLS: B = (X'X)^{-1} X'Y
    XtX = X.T @ X
    XtY = X.T @ Y
    try:
        coefficients = np.linalg.solve(XtX, XtY).T  # shape (k, k*p + 1)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(X, Y, rcond=None)[0].T

    residuals = Y - X @ coefficients.T
    last_values = data[-p:]  # Store last p observations for forecasting

    return VARResult(
        coefficients=coefficients,
        target_cols=target_cols,
        p=p,
        residuals=residuals,
        _last_values=last_values,
    )


def var_forecast(
    model: VARResult,
    horizon: int,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Produce multi-step VAR forecasts.

    Parameters
    ----------
    model
        A fitted :class:`VARResult`.
    horizon
        Number of steps to forecast.
    time_col
        Column name for the step index in the output.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[time_col, target_col_1, ..., target_col_k]``
        where each target column contains the ``"y_hat"`` forecast.

    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    p = model.p
    B = model.coefficients  # (k, k*p + 1)

    # Initialize with last p observations
    history = list(model._last_values)

    forecasts: list[np.ndarray] = []
    for _step in range(horizon):
        row = []
        for lag in range(1, p + 1):
            idx = len(history) - lag
            row.extend(history[idx])
        row.append(1.0)
        x = np.array(row)
        pred = B @ x
        forecasts.append(pred)
        history.append(pred)

    # Build output DataFrame
    result_data: dict[str, Any] = {time_col: list(range(1, horizon + 1))}
    fc_arr = np.array(forecasts)
    for i, col_name in enumerate(model.target_cols):
        result_data[col_name] = fc_arr[:, i].tolist()

    return pl.DataFrame(result_data)


def granger_causality(
    df: pl.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 5,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Test whether *cause_col* Granger-causes *effect_col*.

    Uses an F-test comparing a restricted model (effect's own lags only)
    to an unrestricted model (effect's lags + cause's lags).

    Parameters
    ----------
    df
        Input DataFrame.
    cause_col
        Potential causal variable.
    effect_col
        Variable to predict.
    max_lag
        Maximum lag to test.
    time_col
        Column with timestamps.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["lag", "f_stat", "p_value"]``.

    """
    from scipy.stats import f as f_dist

    sorted_df = df.sort(time_col)
    y = sorted_df[effect_col].to_numpy().astype(np.float64)
    x = sorted_df[cause_col].to_numpy().astype(np.float64)
    n = len(y)

    rows: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        if n <= 2 * lag + 1:
            break

        # Restricted model: y_t ~ y_{t-1} + ... + y_{t-lag} + intercept
        Y = y[lag:]
        X_r = np.column_stack([y[lag - i - 1 : n - i - 1] for i in range(lag)] + [np.ones(n - lag)])

        # Unrestricted: add cause lags
        X_u = np.column_stack(
            [y[lag - i - 1 : n - i - 1] for i in range(lag)]
            + [x[lag - i - 1 : n - i - 1] for i in range(lag)]
            + [np.ones(n - lag)]
        )

        # OLS for both
        rss_r = float(np.sum((Y - X_r @ np.linalg.lstsq(X_r, Y, rcond=None)[0]) ** 2))
        rss_u = float(np.sum((Y - X_u @ np.linalg.lstsq(X_u, Y, rcond=None)[0]) ** 2))

        df_diff = lag  # additional parameters
        df_resid = n - lag - X_u.shape[1]

        if df_resid <= 0 or rss_u <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)
        p_value = 1.0 - f_dist.cdf(f_stat, df_diff, df_resid)

        rows.append({"lag": lag, "f_stat": float(f_stat), "p_value": float(p_value)})

    return pl.DataFrame(rows)
