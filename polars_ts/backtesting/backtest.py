"""Unified backtesting pipeline for time series forecasting models.

Runs a model through cross-validation folds, collects per-fold metrics,
and provides aggregated summaries and optional per-horizon breakdowns.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CVSplitter = Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]
MetricFn = Callable[..., pl.DataFrame | float]


def _has_fit_predict(obj: Any) -> bool:
    return callable(getattr(obj, "fit", None)) and callable(getattr(obj, "predict", None))


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------


def _evaluate_fold(
    model: Any,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    h: int,
    metrics: dict[str, MetricFn],
    actual_col: str,
    predicted_col: str,
    id_col: str,
    time_col: str,
) -> dict[str, float]:
    """Fit a model on *train_df*, predict on *test_df*, compute metrics."""
    fitted = copy.deepcopy(model)
    fitted.fit(train_df)
    preds = fitted.predict(train_df, h=h)

    # Align predictions with actuals
    if "y_hat" in preds.columns and predicted_col != "y_hat":
        preds = preds.rename({"y_hat": predicted_col})
    joined = test_df.select(id_col, time_col, actual_col).join(
        preds,
        on=[id_col, time_col],
        how="inner",
    )

    if joined.is_empty():
        return {name: float("nan") for name in metrics}

    scores: dict[str, float] = {}
    for name, fn in metrics.items():
        val = fn(joined, actual_col=actual_col, predicted_col=predicted_col)
        if isinstance(val, pl.DataFrame):
            scores[name] = float(val.select(pl.exclude(id_col)).mean().row(0)[0])
        else:
            scores[name] = float(val)
    return scores


# ---------------------------------------------------------------------------
# Per-horizon evaluation
# ---------------------------------------------------------------------------


def _per_horizon_scores(
    model: Any,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    h: int,
    metrics: dict[str, MetricFn],
    actual_col: str,
    predicted_col: str,
    id_col: str,
    time_col: str,
) -> list[dict[str, Any]]:
    """Return one row per horizon step with metric scores."""
    fitted = copy.deepcopy(model)
    fitted.fit(train_df)
    preds = fitted.predict(train_df, h=h)

    if "y_hat" in preds.columns and predicted_col != "y_hat":
        preds = preds.rename({"y_hat": predicted_col})
    joined = test_df.select(id_col, time_col, actual_col).join(
        preds,
        on=[id_col, time_col],
        how="inner",
    )

    if joined.is_empty():
        return []

    # Assign horizon step per series based on time ordering
    joined = joined.with_columns(
        pl.col(time_col).rank("ordinal").over(id_col).cast(pl.Int32).alias("horizon_step"),
    )

    rows: list[dict[str, Any]] = []
    for step in sorted(joined["horizon_step"].unique().to_list()):
        step_df = joined.filter(pl.col("horizon_step") == step)
        row: dict[str, Any] = {"horizon_step": int(step)}
        for name, fn in metrics.items():
            val = fn(step_df, actual_col=actual_col, predicted_col=predicted_col)
            if isinstance(val, pl.DataFrame):
                row[name] = float(val.select(pl.exclude(id_col)).mean().row(0)[0])
            else:
                row[name] = float(val)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def backtest(
    model: Any,
    cv: CVSplitter,
    metrics: dict[str, MetricFn],
    *,
    h: int | None = None,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
    n_jobs: int = 1,
    return_predictions: bool = False,
    per_horizon: bool = False,
) -> dict[str, pl.DataFrame]:
    """Run a model through cross-validation folds and collect metrics.

    Parameters
    ----------
    model
        Any object with ``fit(df)`` and ``predict(df, h=...)`` methods
        (e.g. ``ForecastPipeline``, ``GlobalForecaster``).
    cv
        A cross-validation generator yielding ``(train_df, test_df)`` tuples.
        Use ``expanding_window_cv``, ``sliding_window_cv``, or
        ``rolling_origin_cv``.
    metrics
        Mapping of metric name to callable. Each callable must accept
        ``(df, actual_col=, predicted_col=)`` and return a float or
        per-series DataFrame.
    h
        Forecast horizon. If ``None``, inferred from the first test fold.
    actual_col
        Column with actual values.
    predicted_col
        Column name for predictions (internal use).
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    n_jobs
        Number of parallel workers for fold evaluation. ``1`` (default)
        runs sequentially.
    return_predictions
        If ``True``, include a ``"predictions"`` key with per-fold
        forecasts concatenated.
    per_horizon
        If ``True``, include a ``"per_horizon"`` key with metric
        breakdowns by forecast step.

    Returns
    -------
    dict[str, pl.DataFrame]
        Always contains:

        - ``"fold_scores"`` — one row per fold with metric columns.
        - ``"summary"`` — mean and std of each metric across folds.

        Optionally:

        - ``"per_horizon"`` — metric scores broken down by horizon step.
        - ``"predictions"`` — concatenated predictions with fold column.

    """
    if not _has_fit_predict(model):
        raise TypeError("model must have fit(df) and predict(df, h=...) methods")
    if not metrics:
        raise ValueError("metrics must be a non-empty dict")

    folds = list(cv)
    if not folds:
        raise ValueError("cv produced no folds")

    # Infer horizon from first test fold
    if h is None:
        first_test = folds[0][1]
        h = first_test.select(pl.col(time_col).n_unique()).item()

    fold_rows: list[dict[str, Any]] = []
    all_preds: list[pl.DataFrame] = []
    all_horizon_rows: list[dict[str, Any]] = []

    if n_jobs > 1:
        # Parallel execution across folds
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_fold,
                    model,
                    train_df,
                    test_df,
                    h,
                    metrics,
                    actual_col,
                    predicted_col,
                    id_col,
                    time_col,
                )
                for train_df, test_df in folds
            ]
            for i, future in enumerate(futures):
                scores = future.result()
                fold_rows.append({"fold": i, **scores})
    else:
        for i, (train_df, test_df) in enumerate(folds):
            scores = _evaluate_fold(
                model,
                train_df,
                test_df,
                h,
                metrics,
                actual_col,
                predicted_col,
                id_col,
                time_col,
            )
            fold_rows.append({"fold": i, **scores})

            if return_predictions:
                fitted = copy.deepcopy(model)
                fitted.fit(train_df)
                preds = fitted.predict(train_df, h=h).with_columns(pl.lit(i).alias("fold"))
                all_preds.append(preds)

            if per_horizon:
                horizon_rows = _per_horizon_scores(
                    model,
                    train_df,
                    test_df,
                    h,
                    metrics,
                    actual_col,
                    predicted_col,
                    id_col,
                    time_col,
                )
                for row in horizon_rows:
                    row["fold"] = i
                all_horizon_rows.extend(horizon_rows)

    # Build result DataFrames
    fold_scores = pl.DataFrame(fold_rows)

    metric_names = list(metrics.keys())
    summary_rows: list[dict[str, Any]] = []
    for name in metric_names:
        col = fold_scores[name]
        summary_rows.append(
            {
                "metric": name,
                "mean": col.mean(),
                "std": col.std(),
            }
        )
    summary = pl.DataFrame(summary_rows)

    result: dict[str, pl.DataFrame] = {
        "fold_scores": fold_scores,
        "summary": summary,
    }

    if return_predictions and all_preds:
        result["predictions"] = pl.concat(all_preds)

    if per_horizon and all_horizon_rows:
        horizon_df = pl.DataFrame(all_horizon_rows)
        # Aggregate per horizon step across folds
        agg_exprs = [pl.col(name).mean().alias(f"{name}_mean") for name in metric_names] + [
            pl.col(name).std().alias(f"{name}_std") for name in metric_names
        ]
        per_horizon_summary = horizon_df.group_by("horizon_step").agg(agg_exprs).sort("horizon_step")
        result["per_horizon"] = per_horizon_summary

    return result


def compare_models(
    models: dict[str, Any],
    df: pl.DataFrame,
    cv: Callable[..., CVSplitter],
    cv_kwargs: dict[str, Any],
    metrics: dict[str, MetricFn],
    *,
    h: int | None = None,
    actual_col: str = "y",
    predicted_col: str = "y_hat",
    id_col: str = "unique_id",
    time_col: str = "ds",
    n_jobs: int = 1,
) -> dict[str, pl.DataFrame]:
    """Compare multiple models using the same cross-validation setup.

    Parameters
    ----------
    models
        Mapping of model name to model object.
    df
        Full dataset.
    cv
        A CV splitter *function* (not generator) such as
        ``expanding_window_cv``. Called once per model with ``df``
        and ``**cv_kwargs``.
    cv_kwargs
        Keyword arguments passed to ``cv(df, **cv_kwargs)``.
    metrics
        Metric name to callable mapping.
    h
        Forecast horizon (inferred from folds if ``None``).
    actual_col
        Column with actual values.
    predicted_col
        Column name for predictions.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    n_jobs
        Number of parallel workers per model.

    Returns
    -------
    dict[str, pl.DataFrame]
        - ``"comparison"`` — one row per model with mean metric scores.
        - ``"fold_scores"`` — per-fold scores for all models (with
          ``model`` column).

    """
    if not models:
        raise ValueError("models must be a non-empty dict")

    all_fold_rows: list[pl.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        result = backtest(
            model=model,
            cv=cv(df, **cv_kwargs),
            metrics=metrics,
            h=h,
            actual_col=actual_col,
            predicted_col=predicted_col,
            id_col=id_col,
            time_col=time_col,
            n_jobs=n_jobs,
        )

        # Tag fold scores with model name
        fold_df = result["fold_scores"].with_columns(pl.lit(model_name).alias("model"))
        all_fold_rows.append(fold_df)

        # Build comparison row
        summary = result["summary"]
        row: dict[str, Any] = {"model": model_name}
        for metric_row in summary.iter_rows(named=True):
            name = metric_row["metric"]
            row[f"{name}_mean"] = metric_row["mean"]
            row[f"{name}_std"] = metric_row["std"]
        comparison_rows.append(row)

    return {
        "comparison": pl.DataFrame(comparison_rows),
        "fold_scores": pl.concat(all_fold_rows),
    }
