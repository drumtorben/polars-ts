"""Residual diagnostics for forecast model evaluation. Closes #57."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def acf(
    df: pl.DataFrame,
    target_col: str = "y",
    max_lags: int = 20,
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Compute autocorrelation function with confidence bands.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to compute ACF on.
    max_lags
        Maximum number of lags.
    id_col
        Column identifying each time series.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "lag", "acf", "ci_lower", "ci_upper"]``.
        Confidence bands use the 95% level (``±1.96/√n``).

    """
    if max_lags < 1:
        raise ValueError("max_lags must be >= 1")

    rows: list[dict[str, Any]] = []
    for group_id, group_df in df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        values = np.array(group_df[target_col].drop_nulls().to_list(), dtype=np.float64)
        n = len(values)
        if n < 2:
            continue

        mean = values.mean()
        demeaned = values - mean
        var = float(np.dot(demeaned, demeaned))
        ci = 1.96 / np.sqrt(n)

        for lag in range(0, min(max_lags + 1, n)):
            if var == 0:
                acf_val = 1.0 if lag == 0 else 0.0
            else:
                acf_val = float(np.dot(demeaned[: n - lag], demeaned[lag:])) / var
            rows.append(
                {
                    id_col: gid,
                    "lag": lag,
                    "acf": acf_val,
                    "ci_lower": -ci,
                    "ci_upper": ci,
                }
            )

    return pl.DataFrame(rows)


def pacf(
    df: pl.DataFrame,
    target_col: str = "y",
    max_lags: int = 20,
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Compute partial autocorrelation function via Durbin-Levinson.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to compute PACF on.
    max_lags
        Maximum number of lags.
    id_col
        Column identifying each time series.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "lag", "pacf", "ci_lower", "ci_upper"]``.

    """
    if max_lags < 1:
        raise ValueError("max_lags must be >= 1")

    # First compute ACF values, then derive PACF via Durbin-Levinson
    rows: list[dict[str, Any]] = []
    for group_id, group_df in df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        values = np.array(group_df[target_col].drop_nulls().to_list(), dtype=np.float64)
        n = len(values)
        if n < 2:
            continue

        mean = values.mean()
        demeaned = values - mean
        var = float(np.dot(demeaned, demeaned))
        ci = 1.96 / np.sqrt(n)

        # Compute ACF values
        k = min(max_lags + 1, n)
        acf_vals = np.zeros(k)
        for lag in range(k):
            if var == 0:
                acf_vals[lag] = 1.0 if lag == 0 else 0.0
            else:
                acf_vals[lag] = float(np.dot(demeaned[: n - lag], demeaned[lag:])) / var

        # Durbin-Levinson recursion for PACF
        rows.append({id_col: gid, "lag": 0, "pacf": 1.0, "ci_lower": -ci, "ci_upper": ci})

        if k > 1:
            phi = np.zeros((k, k))
            phi[1, 1] = acf_vals[1]
            rows.append({id_col: gid, "lag": 1, "pacf": float(phi[1, 1]), "ci_lower": -ci, "ci_upper": ci})

            for m in range(2, k):
                num = acf_vals[m] - sum(phi[m - 1, j] * acf_vals[m - j] for j in range(1, m))
                den = 1.0 - sum(phi[m - 1, j] * acf_vals[j] for j in range(1, m))
                if abs(den) < 1e-15:
                    phi[m, m] = 0.0
                else:
                    phi[m, m] = num / den
                for j in range(1, m):
                    phi[m, j] = phi[m - 1, j] - phi[m, m] * phi[m - 1, m - j]
                rows.append({id_col: gid, "lag": m, "pacf": float(phi[m, m]), "ci_lower": -ci, "ci_upper": ci})

    return pl.DataFrame(rows)


def ljung_box(
    df: pl.DataFrame,
    target_col: str = "y",
    lags: list[int] | None = None,
    id_col: str = "unique_id",
) -> pl.DataFrame:
    """Ljung-Box test for residual autocorrelation.

    Parameters
    ----------
    df
        Input DataFrame (typically residuals).
    target_col
        Column to test.
    lags
        Lag values at which to compute the test. Defaults to
        ``[10, 20]``.
    id_col
        Column identifying each time series.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "lag", "q_stat", "p_value"]``.

    """
    from scipy.stats import chi2

    if lags is None:
        lags = [10, 20]

    rows: list[dict[str, Any]] = []
    for group_id, group_df in df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        values = np.array(group_df[target_col].drop_nulls().to_list(), dtype=np.float64)
        n = len(values)
        if n < 3:
            continue

        mean = values.mean()
        demeaned = values - mean
        var = float(np.dot(demeaned, demeaned))

        for max_lag in lags:
            if max_lag >= n:
                continue
            q = 0.0
            for k in range(1, max_lag + 1):
                if var == 0:
                    rk = 0.0
                else:
                    rk = float(np.dot(demeaned[: n - k], demeaned[k:])) / var
                q += rk**2 / (n - k)
            q *= n * (n + 2)
            p_value = 1.0 - chi2.cdf(q, df=max_lag)
            rows.append({id_col: gid, "lag": max_lag, "q_stat": q, "p_value": p_value})

    return pl.DataFrame(rows)
