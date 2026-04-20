"""PELT (Pruned Exact Linear Time) changepoint detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _cost_mean(data: np.ndarray, start: int, end: int) -> float:
    """Cost of segment [start, end) under a change-in-mean model."""
    seg = data[start:end]
    if len(seg) == 0:
        return 0.0
    return float(np.sum((seg - seg.mean()) ** 2))


def _cost_var(data: np.ndarray, start: int, end: int) -> float:
    """Cost of segment [start, end) under a change-in-variance model."""
    seg = data[start:end]
    n = len(seg)
    if n < 2:
        return 0.0
    var = float(np.var(seg, ddof=1))
    if var <= 0:
        return 0.0
    return n * np.log(var)


def _cost_meanvar(data: np.ndarray, start: int, end: int) -> float:
    """Cost of segment under a change-in-mean-and-variance model."""
    return _cost_mean(data, start, end) + _cost_var(data, start, end)


_COST_FNS = {"mean": _cost_mean, "var": _cost_var, "meanvar": _cost_meanvar}


def pelt(
    df: pl.DataFrame,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    cost: str = "mean",
    penalty: float | None = None,
    min_size: int = 2,
) -> pl.DataFrame:
    """Detect multiple changepoints using the PELT algorithm.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to analyze.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    cost
        Cost function: ``"mean"``, ``"var"``, or ``"meanvar"``.
    penalty
        Penalty per changepoint. Defaults to ``2 * log(n)`` (BIC-like).
    min_size
        Minimum segment length between changepoints.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "changepoint_idx", time_col]``
        listing detected changepoint locations.

    """
    if cost not in _COST_FNS:
        raise ValueError(f"Unknown cost {cost!r}. Choose from {sorted(_COST_FNS)}")

    cost_fn = _COST_FNS[cost]
    sorted_df = df.sort(id_col, time_col)

    rows: list[dict[str, Any]] = []
    for group_id, group_df in sorted_df.group_by(id_col, maintain_order=True):
        gid = group_id[0]
        data = np.array(group_df[target_col].to_list(), dtype=np.float64)
        times = group_df[time_col].to_list()
        n = len(data)

        pen = penalty if penalty is not None else 2 * np.log(n)

        # PELT dynamic programming
        # F[t] = min cost of segmenting data[0:t]
        f = np.full(n + 1, np.inf)
        f[0] = -pen
        cp: list[list[int]] = [[] for _ in range(n + 1)]
        candidates = [0]

        for t in range(min_size, n + 1):
            best_cost = np.inf
            best_s = 0
            for s in candidates:
                if t - s >= min_size:
                    c = f[s] + cost_fn(data, s, t) + pen
                    if c < best_cost:
                        best_cost = c
                        best_s = s
            f[t] = best_cost
            cp[t] = cp[best_s] + [best_s]

            # Pruning: remove candidates that can never be optimal
            candidates = [s for s in candidates if f[s] + cost_fn(data, s, t) <= f[t]] + [t]

        # Extract changepoints (exclude 0)
        changepoints = [c for c in cp[n] if c > 0]

        for idx in changepoints:
            rows.append({id_col: gid, "changepoint_idx": idx, time_col: times[idx]})

    if not rows:
        schema = {id_col: df.schema[id_col], "changepoint_idx": pl.Int64(), time_col: df.schema[time_col]}
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows)
