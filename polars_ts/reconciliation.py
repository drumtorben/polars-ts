"""Forecast reconciliation for hierarchical time series. Closes #55."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def reconcile(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    method: str = "bottom_up",
    forecast_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Reconcile forecasts across a hierarchy so they sum coherently.

    Parameters
    ----------
    df
        DataFrame with forecasts at all levels, identified by ``id_col``.
    hierarchy
        Mapping from child node to parent node (e.g.
        ``{"product_A": "category_1", "product_B": "category_1",
          "category_1": "total"}``).
    method
        Reconciliation method: ``"bottom_up"``, ``"top_down"``, or
        ``"ols"`` (MinTrace-OLS).
    forecast_col
        Column with forecast values.
    id_col
        Column identifying each node in the hierarchy.
    time_col
        Column with timestamps.

    Returns
    -------
    pl.DataFrame
        Reconciled forecasts with the same schema as *df*.

    """
    valid_methods = {"bottom_up", "top_down", "ols"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid_methods)}")

    if method == "bottom_up":
        return _bottom_up(df, hierarchy, forecast_col, id_col, time_col)
    if method == "top_down":
        return _top_down(df, hierarchy, forecast_col, id_col, time_col)
    return _ols(df, hierarchy, forecast_col, id_col, time_col)


def _get_bottom_nodes(hierarchy: dict[str, str]) -> list[str]:
    """Return nodes that are not parents of anything (leaf nodes)."""
    parents = set(hierarchy.values())
    children = set(hierarchy.keys())
    return sorted(children - parents)


def _get_top_node(hierarchy: dict[str, str]) -> str:
    """Return the root node (parent that is not a child of anything)."""
    parents = set(hierarchy.values())
    children = set(hierarchy.keys())
    roots = parents - children
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root node, found {roots}")
    return roots.pop()


def _get_children(hierarchy: dict[str, str], parent: str) -> list[str]:
    """Return direct children of a parent node."""
    return sorted(k for k, v in hierarchy.items() if v == parent)


def _bottom_up(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Aggregate bottom-level forecasts upward."""
    bottom = _get_bottom_nodes(hierarchy)
    bottom_df = df.filter(pl.col(id_col).is_in(bottom))

    # Build aggregation levels
    result_frames = [bottom_df]

    # Aggregate upward through hierarchy
    processed: set[str] = set(bottom)

    while True:
        # Find parents whose all children are processed
        parent_groups: dict[str, list[str]] = {}
        for child, parent in hierarchy.items():
            if child in processed and parent not in processed:
                parent_groups.setdefault(parent, []).append(child)

        if not parent_groups:
            break

        for parent, _children in parent_groups.items():
            # Check all children of this parent are processed
            all_children = _get_children(hierarchy, parent)
            if not all(c in processed for c in all_children):
                continue

            child_data = df.filter(pl.col(id_col).is_in(all_children))
            if parent in processed:
                continue

            # Sum children forecasts
            agg = (
                child_data.group_by(time_col).agg(pl.col(forecast_col).sum()).with_columns(pl.lit(parent).alias(id_col))
            )
            # Reorder columns
            agg = agg.select(id_col, time_col, forecast_col)
            result_frames.append(agg)
            processed.add(parent)

            # Update child_data for next level
            df = pl.concat([df, agg])

    return pl.concat(result_frames).sort(id_col, time_col)


def _top_down(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Disaggregate top-level forecast using historical proportions."""
    top = _get_top_node(hierarchy)
    bottom = _get_bottom_nodes(hierarchy)

    top_forecasts = df.filter(pl.col(id_col) == top)
    bottom_actuals = df.filter(pl.col(id_col).is_in(bottom))

    # Compute proportions from bottom-level data
    total = bottom_actuals.group_by(time_col).agg(pl.col(forecast_col).sum().alias("__total"))
    with_total = bottom_actuals.join(total, on=time_col)
    proportions = with_total.group_by(id_col).agg((pl.col(forecast_col) / pl.col("__total")).mean().alias("__prop"))

    # Disaggregate
    result_frames = [top_forecasts]
    for row in proportions.iter_rows(named=True):
        node_id = row[id_col]
        prop = row["__prop"]
        node_fc = top_forecasts.with_columns(
            (pl.col(forecast_col) * prop).alias(forecast_col),
            pl.lit(node_id).alias(id_col),
        )
        result_frames.append(node_fc)

    # Build intermediate levels
    result = pl.concat(result_frames)
    processed = set(bottom) | {top}

    for _child, parent in hierarchy.items():
        if parent not in processed and parent != top:
            children = _get_children(hierarchy, parent)
            child_data = result.filter(pl.col(id_col).is_in(children))
            agg = (
                child_data.group_by(time_col)
                .agg(pl.col(forecast_col).sum())
                .with_columns(pl.lit(parent).alias(id_col))
                .select(id_col, time_col, forecast_col)
            )
            result = pl.concat([result, agg])
            processed.add(parent)

    return result.sort(id_col, time_col)


def _ols(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """MinTrace-OLS reconciliation.

    Computes reconciled forecasts as: y_tilde = S @ (S'S)^{-1} @ S' @ y_hat
    where S is the summing matrix.
    """
    all_nodes = sorted(set(hierarchy.keys()) | set(hierarchy.values()))
    bottom = _get_bottom_nodes(hierarchy)
    n_total = len(all_nodes)
    n_bottom = len(bottom)
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    # Build summing matrix S: maps bottom-level to all levels
    S = np.zeros((n_total, n_bottom))
    for j, b in enumerate(bottom):
        # Trace from bottom to top
        current = b
        S[node_idx[current], j] = 1.0
        while current in hierarchy:
            current = hierarchy[current]
            S[node_idx[current], j] = 1.0

    # Reconciliation: P = S @ (S'S)^{-1} @ S'
    StS_inv = np.linalg.pinv(S.T @ S)
    P = S @ StS_inv @ S.T

    # Apply per time step
    times = sorted(df[time_col].unique().to_list())
    result_rows: list[dict[str, Any]] = []

    for t in times:
        t_data = df.filter(pl.col(time_col) == t)
        y_hat = np.zeros(n_total)
        for row in t_data.iter_rows(named=True):
            if row[id_col] in node_idx:
                y_hat[node_idx[row[id_col]]] = row[forecast_col]

        y_tilde = P @ y_hat

        for node, idx in node_idx.items():
            result_rows.append({id_col: node, time_col: t, forecast_col: y_tilde[idx]})

    return pl.DataFrame(result_rows).sort(id_col, time_col)
