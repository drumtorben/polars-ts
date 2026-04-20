"""Isolation Forest adapter for time series anomaly detection. Closes #63."""

from __future__ import annotations

from typing import Any

import polars as pl


def isolation_forest_detect(
    df: pl.DataFrame,
    feature_cols: list[str],
    id_col: str = "unique_id",
    time_col: str = "ds",
    contamination: float = 0.05,
    global_model: bool = True,
    seed: int = 42,
    **kwargs: Any,
) -> pl.DataFrame:
    """Detect anomalies using Isolation Forest on engineered features.

    Requires scikit-learn (``pip install polars-timeseries[ml]``).

    Parameters
    ----------
    df
        Input DataFrame with feature columns already computed
        (e.g. via lag_features, rolling_features, calendar_features).
    feature_cols
        Column names to use as features for the Isolation Forest.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    contamination
        Expected proportion of outliers.
    global_model
        If ``True``, fit a single model on all series.
        If ``False``, fit a separate model per series.
    seed
        Random seed.
    **kwargs
        Additional arguments passed to ``IsolationForest``.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``"anomaly_score"`` (float, lower = more anomalous)
        and ``"is_anomaly"`` (boolean) columns appended.

    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError(
            "scikit-learn is required for isolation_forest_detect. " "Install with: pip install polars-timeseries[ml]"
        ) from None

    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    sorted_df = df.sort(id_col, time_col)

    # Drop nulls in feature columns for fitting
    clean = sorted_df.drop_nulls(subset=feature_cols)

    if global_model:
        X = clean.select(feature_cols).to_numpy()
        iso = IsolationForest(contamination=contamination, random_state=seed, **kwargs)
        iso.fit(X)
        scores = iso.decision_function(X)
        labels = iso.predict(X)
        return clean.with_columns(
            pl.Series("anomaly_score", scores.tolist()),
            pl.Series("is_anomaly", (labels == -1).tolist()),
        )

    # Per-series
    result_frames: list[pl.DataFrame] = []
    for _gid, group_df in clean.group_by(id_col, maintain_order=True):
        X = group_df.select(feature_cols).to_numpy()
        if len(X) < 2:
            result_frames.append(
                group_df.with_columns(
                    pl.lit(0.0).alias("anomaly_score"),
                    pl.lit(False).alias("is_anomaly"),
                )
            )
            continue
        iso = IsolationForest(contamination=contamination, random_state=seed, **kwargs)
        iso.fit(X)
        scores = iso.decision_function(X)
        labels = iso.predict(X)
        result_frames.append(
            group_df.with_columns(
                pl.Series("anomaly_score", scores.tolist()),
                pl.Series("is_anomaly", (labels == -1).tolist()),
            )
        )

    return pl.concat(result_frames)
