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
    *,
    middle_level: list[str] | None = None,
    residuals: pl.DataFrame | None = None,
    train_data: pl.DataFrame | None = None,
    n_folds: int = 5,
    interval_cols: list[str] | None = None,
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
        Reconciliation method: ``"bottom_up"``, ``"top_down"``, ``"ols"``
        (MinTrace-OLS), ``"middle_out"``, ``"permbu"``, or ``"mint_cv"``
        (MinTrace with cross-validation).
    forecast_col
        Column with forecast values.
    id_col
        Column identifying each node in the hierarchy.
    time_col
        Column with timestamps.
    middle_level
        Node names for the anchor level (required for ``"middle_out"``).
    residuals
        DataFrame with columns ``[id_col, time_col, "residual"]`` containing
        historical forecast residuals (required for ``"permbu"``).
    train_data
        Historical forecasts for cross-validation (required for ``"mint_cv"``).
    n_folds
        Number of CV folds for ``"mint_cv"`` (default 5).
    interval_cols
        Column names for prediction intervals (e.g. ``["y_lower", "y_upper"]``).
        When provided, each interval column is reconciled independently using
        the same projection matrix as the point forecasts.

    Returns
    -------
    pl.DataFrame
        Reconciled forecasts with the same schema as *df*.

    """
    valid_methods = {"bottom_up", "top_down", "ols", "middle_out", "permbu", "mint_cv"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid_methods)}")

    if method == "middle_out":
        if middle_level is None:
            raise ValueError("middle_level is required for method='middle_out'")
        return _middle_out(df, hierarchy, forecast_col, id_col, time_col, middle_level)
    if method == "permbu":
        if residuals is None:
            raise ValueError("residuals is required for method='permbu'")
        return _permbu(df, hierarchy, forecast_col, id_col, time_col, residuals)
    if method == "mint_cv":
        if train_data is None:
            raise ValueError("train_data is required for method='mint_cv'")
        return _mint_cv(df, hierarchy, forecast_col, id_col, time_col, train_data, n_folds)

    # Methods that support interval_cols via projection matrix
    if method == "bottom_up":
        return _bottom_up(df, hierarchy, forecast_col, id_col, time_col)
    if method == "top_down":
        return _top_down(df, hierarchy, forecast_col, id_col, time_col)
    return _ols(df, hierarchy, forecast_col, id_col, time_col, interval_cols=interval_cols)


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


def _build_summing_matrix(
    hierarchy: dict[str, str],
) -> tuple[np.ndarray, list[str], list[str], dict[str, int]]:
    """Build the summing matrix S and return (S, all_nodes, bottom, node_idx)."""
    all_nodes = sorted(set(hierarchy.keys()) | set(hierarchy.values()))
    bottom = _get_bottom_nodes(hierarchy)
    n_total = len(all_nodes)
    n_bottom = len(bottom)
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    S = np.zeros((n_total, n_bottom))
    for j, b in enumerate(bottom):
        current = b
        S[node_idx[current], j] = 1.0
        while current in hierarchy:
            current = hierarchy[current]
            S[node_idx[current], j] = 1.0

    return S, all_nodes, bottom, node_idx


def _apply_projection(
    df: pl.DataFrame,
    P: np.ndarray,
    all_nodes: list[str],
    node_idx: dict[str, int],
    forecast_col: str,
    id_col: str,
    time_col: str,
    extra_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Apply projection matrix P to forecasts, optionally to extra columns too."""
    cols_to_project = [forecast_col] + (extra_cols or [])
    n_total = len(all_nodes)
    times = sorted(df[time_col].unique().to_list())
    result_rows: list[dict[str, Any]] = []

    for t in times:
        t_data = df.filter(pl.col(time_col) == t)
        row_map = {row[id_col]: row for row in t_data.iter_rows(named=True)}

        for col in cols_to_project:
            y_hat = np.zeros(n_total)
            for node, idx in node_idx.items():
                if node in row_map:
                    y_hat[idx] = row_map[node][col]
            if col == cols_to_project[0]:
                y_tildes = {col: P @ y_hat}
            else:
                y_tildes[col] = P @ y_hat

        for node, idx in node_idx.items():
            row_dict: dict[str, Any] = {id_col: node, time_col: t}
            for col in cols_to_project:
                row_dict[col] = y_tildes[col][idx]
            result_rows.append(row_dict)

    return pl.DataFrame(result_rows).sort(id_col, time_col)


def _ols(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
    *,
    interval_cols: list[str] | None = None,
) -> pl.DataFrame:
    """MinTrace-OLS reconciliation.

    Computes reconciled forecasts as: y_tilde = S @ (S'S)^{-1} @ S' @ y_hat
    where S is the summing matrix.
    """
    S, all_nodes, _bottom, node_idx = _build_summing_matrix(hierarchy)

    # Reconciliation: P = S @ (S'S)^{-1} @ S'
    StS_inv = np.linalg.pinv(S.T @ S)
    P = S @ StS_inv @ S.T

    return _apply_projection(df, P, all_nodes, node_idx, forecast_col, id_col, time_col, extra_cols=interval_cols)


def _middle_out(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
    middle_level: list[str],
) -> pl.DataFrame:
    """Middle-out reconciliation: anchor at intermediate level.

    Below the middle level: disaggregate using historical proportions.
    Above the middle level: aggregate by summing children.
    """
    bottom = _get_bottom_nodes(hierarchy)

    # --- Disaggregate downward from middle to bottom ---
    # For each middle node, find its descendant bottom nodes
    def _get_descendant_bottom(node: str) -> list[str]:
        children = _get_children(hierarchy, node)
        if not children:
            return [node] if node in bottom else []
        result: list[str] = []
        for c in children:
            if c in bottom:
                result.append(c)
            else:
                result.extend(_get_descendant_bottom(c))
        return result

    result_frames: list[pl.DataFrame] = []

    # Keep middle-level forecasts as-is
    middle_df = df.filter(pl.col(id_col).is_in(middle_level))
    result_frames.append(middle_df)

    # Disaggregate to bottom using proportions from input forecasts
    for mid_node in middle_level:
        desc_bottom = _get_descendant_bottom(mid_node)
        if not desc_bottom:
            continue

        bottom_data = df.filter(pl.col(id_col).is_in(desc_bottom))
        mid_data = df.filter(pl.col(id_col) == mid_node)

        # Compute proportions per time step, then average
        total_per_t = bottom_data.group_by(time_col).agg(pl.col(forecast_col).sum().alias("__total"))
        with_total = bottom_data.join(total_per_t, on=time_col)
        proportions = with_total.group_by(id_col).agg((pl.col(forecast_col) / pl.col("__total")).mean().alias("__prop"))

        for row in proportions.iter_rows(named=True):
            node_id = row[id_col]
            prop = row["__prop"]
            node_fc = mid_data.with_columns(
                (pl.col(forecast_col) * prop).alias(forecast_col),
                pl.lit(node_id).alias(id_col),
            )
            result_frames.append(node_fc)

    # --- Aggregate upward from middle to top ---
    result = pl.concat(result_frames)
    processed = set(bottom) | set(middle_level)

    # Build intermediate nodes between middle and top
    # First aggregate middle → parents, then parents → grandparents, etc.
    while True:
        new_parents: dict[str, list[str]] = {}
        for child, parent in hierarchy.items():
            if child in processed and parent not in processed:
                new_parents.setdefault(parent, []).append(child)

        if not new_parents:
            break

        for parent, _children in new_parents.items():
            all_children = _get_children(hierarchy, parent)
            if not all(c in processed for c in all_children):
                continue

            child_data = result.filter(pl.col(id_col).is_in(all_children))
            agg = (
                child_data.group_by(time_col)
                .agg(pl.col(forecast_col).sum())
                .with_columns(pl.lit(parent).alias(id_col))
                .select(id_col, time_col, forecast_col)
            )
            result = pl.concat([result, agg])
            processed.add(parent)

    return result.sort(id_col, time_col)


def _permbu(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
    residuals: pl.DataFrame,
) -> pl.DataFrame:
    """PERMBU: Projection-based Empirical Residual MinTrace Bottom-Up.

    Uses empirical residual covariance to weight the MinTrace reconciliation,
    producing a shrinkage estimator between OLS and sample-covariance MinTrace.
    """
    S, all_nodes, bottom, node_idx = _build_summing_matrix(hierarchy)
    n_total = len(all_nodes)

    # Build residual matrix from historical data
    res_times = sorted(residuals[time_col].unique().to_list())
    E = np.zeros((n_total, len(res_times)))
    for t_idx, t in enumerate(res_times):
        t_data = residuals.filter(pl.col(time_col) == t)
        for row in t_data.iter_rows(named=True):
            if row[id_col] in node_idx:
                E[node_idx[row[id_col]], t_idx] = row["residual"]

    # Empirical covariance W = (1/T) E @ E'
    T = len(res_times)
    W = (E @ E.T) / T

    # Regularize: shrink toward diagonal
    W_diag = np.diag(np.diag(W))
    alpha = 0.5
    W_reg = alpha * W_diag + (1 - alpha) * W

    # MinTrace with covariance: P = S @ (S' W^{-1} S)^{-1} @ S' @ W^{-1}
    W_inv = np.linalg.pinv(W_reg)
    StWinvS_inv = np.linalg.pinv(S.T @ W_inv @ S)
    P = S @ StWinvS_inv @ S.T @ W_inv

    return _apply_projection(df, P, all_nodes, node_idx, forecast_col, id_col, time_col)


def _mint_cv(
    df: pl.DataFrame,
    hierarchy: dict[str, str],
    forecast_col: str,
    id_col: str,
    time_col: str,
    train_data: pl.DataFrame,
    n_folds: int,
) -> pl.DataFrame:
    """MinTrace with cross-validation for covariance estimation.

    Splits train_data into folds, computes in-sample residuals per fold,
    and averages the resulting covariance matrices for a robust W estimate.
    """
    S, all_nodes, bottom, node_idx = _build_summing_matrix(hierarchy)
    n_total = len(all_nodes)

    train_times = sorted(train_data[time_col].unique().to_list())
    n_times = len(train_times)
    actual_folds = min(n_folds, n_times)

    # Build forecast matrix from train_data
    Y = np.zeros((n_total, n_times))
    for t_idx, t in enumerate(train_times):
        t_data = train_data.filter(pl.col(time_col) == t)
        for row in t_data.iter_rows(named=True):
            if row[id_col] in node_idx:
                Y[node_idx[row[id_col]], t_idx] = row[forecast_col]

    # Cross-validated covariance: average across folds
    fold_size = n_times // actual_folds
    W_sum = np.zeros((n_total, n_total))

    for fold in range(actual_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < actual_folds - 1 else n_times
        train_idx = list(range(0, val_start)) + list(range(val_end, n_times))

        if not train_idx:
            continue

        # Compute mean forecast per node on train split
        Y_train = Y[:, train_idx]
        mean_forecast = Y_train.mean(axis=1, keepdims=True)

        # Residuals = actuals - mean (on validation split)
        Y_val = Y[:, val_start:val_end]
        residuals_fold = Y_val - mean_forecast

        # Fold covariance
        W_fold = (residuals_fold @ residuals_fold.T) / residuals_fold.shape[1]
        W_sum += W_fold

    W = W_sum / actual_folds

    # Regularize toward diagonal
    W_diag = np.diag(np.diag(W))
    lam = 0.5
    W_reg = lam * W_diag + (1 - lam) * W

    # MinTrace: P = S @ (S' W^{-1} S)^{-1} @ S' @ W^{-1}
    W_inv = np.linalg.pinv(W_reg)
    StWinvS_inv = np.linalg.pinv(S.T @ W_inv @ S)
    P = S @ StWinvS_inv @ S.T @ W_inv

    return _apply_projection(df, P, all_nodes, node_idx, forecast_col, id_col, time_col)
