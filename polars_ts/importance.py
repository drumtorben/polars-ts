"""Permutation-based feature importance for time series. Closes #59."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl


def permutation_importance(
    df: pl.DataFrame,
    model: Any,
    feature_cols: list[str],
    target_col: str = "y",
    metric_fn: Callable[[pl.DataFrame], float] | None = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Compute permutation importance for each feature.

    Shuffles each feature column and measures degradation in the
    scoring metric. Higher importance means the model relies more
    on that feature.

    Parameters
    ----------
    df
        Evaluation DataFrame with features and target.
    model
        A fitted sklearn-compatible estimator with ``predict``.
    feature_cols
        Column names of the features.
    target_col
        Column with actual values.
    metric_fn
        Scoring function ``fn(df_with_y_and_y_hat) -> float``.
        Lower is better (e.g. MAE). Defaults to MAE.
    n_repeats
        Number of permutation repeats per feature.
    seed
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["feature", "importance_mean", "importance_std"]``,
        sorted by importance (descending).

    """
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    if metric_fn is None:
        from polars_ts.metrics.forecast import mae

        def metric_fn(d: pl.DataFrame) -> float:
            score = mae(d, actual_col=target_col, predicted_col="y_hat")
            return float(score) if isinstance(score, float) else float(score.item())  # type: ignore[union-attr]

    rng = np.random.default_rng(seed)

    # Baseline score
    X = df.select(feature_cols).to_numpy().astype(np.float64)
    preds = model.predict(X)
    baseline_df = df.with_columns(pl.Series("y_hat", preds.tolist()))
    baseline_score = metric_fn(baseline_df)

    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        scores: list[float] = []
        for _ in range(n_repeats):
            # Shuffle this column
            shuffled = df.with_columns(pl.Series(col, rng.permutation(df[col].to_numpy()).tolist()))
            X_perm = shuffled.select(feature_cols).to_numpy().astype(np.float64)
            preds_perm = model.predict(X_perm)
            perm_df = shuffled.with_columns(pl.Series("y_hat", preds_perm.tolist()))
            perm_score = metric_fn(perm_df)
            scores.append(perm_score - baseline_score)

        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    result = pl.DataFrame(rows)
    return result.sort("importance_mean", descending=True)
