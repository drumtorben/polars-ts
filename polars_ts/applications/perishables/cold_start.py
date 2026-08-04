"""Cold-start forecasting for new SKUs with little or no history.

New perishables launch constantly; with days of history there is
nothing for a statistical model to fit. Instead, borrow the early-life
trajectory of established SKUs:

- With some observations, find the *k* donors whose own launch-aligned,
  level-normalized early sales look most similar (Euclidean distance)
  and average their subsequent trajectory, rescaled to the new SKU's
  observed level.
- With zero or near-zero observations, fall back to the mean
  launch-aligned trajectory of the SKU's category (or of all donors).

For richer similarity structure (DTW, k-medoids), cluster donor
profiles with :func:`polars_ts.auto_cluster` upstream and pass a single
cluster's SKUs into this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.models._time_utils import _infer_freq, _make_future_dates

_EPS = 1e-9


def cold_start_skus(
    df: pl.DataFrame,
    min_history: int = 28,
    id_col: str = "unique_id",
) -> list[str]:
    """List SKUs with fewer than ``min_history`` observations.

    Parameters
    ----------
    df
        Gap-free canonical sales frame (per-SKU calendars start at each
        SKU's launch, as produced by the loader).
    min_history
        Observation count below which a SKU is considered cold.
    id_col
        SKU column name.

    Returns
    -------
    list[str]
        Cold SKU identifiers, sorted.

    """
    counts = df.group_by(id_col).agg(pl.len().alias("__n"))
    return sorted(counts.filter(pl.col("__n") < min_history)[id_col].to_list())


def _launch_profiles(df: pl.DataFrame, id_col: str, time_col: str, target_col: str) -> dict[str, np.ndarray]:
    """Return each SKU's launch-aligned demand array, sorted by date."""
    profiles: dict[str, np.ndarray] = {}
    for group_id, group_df in df.sort(id_col, time_col).group_by(id_col, maintain_order=True):
        profiles[group_id[0]] = group_df[target_col].to_numpy().astype(np.float64)
    return profiles


def _donor_trajectory(
    obs: np.ndarray,
    donors: list[np.ndarray],
    h: int,
    k: int,
) -> np.ndarray:
    """Forecast h steps by averaging the k most similar donor trajectories.

    Donors are compared on the launch-aligned overlap with ``obs`` after
    mean-normalization, then their steps ``[len(obs), len(obs)+h)`` are
    averaged and rescaled to the new SKU's observed mean level.
    """
    n_obs = len(obs)
    usable = [d for d in donors if len(d) >= n_obs + 1]
    if not usable:
        return np.zeros(h)

    obs_mean = obs.mean() if n_obs else 0.0
    if n_obs and obs_mean > _EPS:
        obs_norm = obs / obs_mean
        dists = []
        for d in usable:
            head = d[:n_obs]
            head_norm = head / max(head.mean(), _EPS)
            dists.append(float(np.linalg.norm(obs_norm - head_norm)))
        order = np.argsort(dists)[:k]
    else:
        # No usable level signal: every donor is equally plausible
        order = np.arange(min(k, len(usable)))

    steps = np.zeros(h)
    for rank in order:
        d = usable[rank]
        future = d[n_obs : n_obs + h]
        head_mean = max(d[:n_obs].mean() if n_obs else d.mean(), _EPS)
        # Express the donor's future relative to its own early level ...
        scaled = future / head_mean
        if len(scaled) < h:
            scaled = np.pad(scaled, (0, h - len(scaled)), mode="edge")
        steps += scaled
    steps /= len(order)
    # ... then rescale to the new SKU's level (donor-average level if unobserved)
    level = (
        obs_mean
        if obs_mean > _EPS
        else float(np.mean([max(d[:n_obs].mean() if n_obs else d.mean(), _EPS) for d in (usable[r] for r in order)]))
    )
    return np.clip(steps * level, 0.0, None)


def cold_start_forecast(
    df: pl.DataFrame,
    h: int,
    min_history: int = 28,
    k: int = 5,
    category_map: pl.DataFrame | None = None,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Forecast cold SKUs by borrowing established SKUs' early life.

    Parameters
    ----------
    df
        Gap-free canonical sales frame containing both cold and
        established SKUs.
    h
        Forecast horizon (periods).
    min_history
        Observation count below which a SKU is treated as cold.
    k
        Number of donor SKUs averaged per cold SKU.
    category_map
        Optional ``[id_col, "category"]`` frame; when given, donors are
        restricted to the cold SKU's category (falling back to all
        donors for unknown categories).
    target_col, id_col, time_col
        Canonical column names.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]`` with *h* rows per cold
        SKU. Established SKUs are not forecast here — route them to the
        standard pipeline.

    """
    if h <= 0:
        raise ValueError("Horizon h must be a positive integer")
    cold = set(cold_start_skus(df, min_history, id_col))
    if not cold:
        schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
        return pl.DataFrame(schema=schema)

    profiles = _launch_profiles(df, id_col, time_col, target_col)
    donor_ids = [sku for sku in profiles if sku not in cold]
    if not donor_ids:
        raise ValueError("No established SKUs available as cold-start donors")

    categories: dict[str, Any] = {}
    if category_map is not None:
        categories = dict(category_map.select(id_col, "category").iter_rows())

    sorted_df = df.sort(id_col, time_col)
    freq = _infer_freq(sorted_df[time_col])
    last_dates = dict(sorted_df.group_by(id_col).agg(pl.col(time_col).max()).iter_rows())

    rows: list[dict[str, Any]] = []
    for sku in sorted(cold):
        cat = categories.get(sku)
        pool_ids = [d for d in donor_ids if categories.get(d) == cat] if cat is not None else donor_ids
        pool = [profiles[d] for d in (pool_ids or donor_ids)]
        y_hat = _donor_trajectory(profiles[sku], pool, h, k)
        for step, t in enumerate(_make_future_dates(last_dates[sku], freq, h)):
            rows.append({id_col: sku, time_col: t, "y_hat": float(y_hat[step])})

    schema = {id_col: df.schema[id_col], time_col: df.schema[time_col], "y_hat": pl.Float64()}
    return pl.DataFrame(rows, schema=schema).sort(id_col, time_col)
