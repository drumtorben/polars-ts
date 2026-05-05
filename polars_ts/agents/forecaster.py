"""ForecasterAgent: model fitting, validation, and adaptive ensemble selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend
from polars_ts.agents.planner import ForecastPlan


@dataclass
class ForecastAgentResult:
    """Output of the ForecasterAgent."""

    predictions: pl.DataFrame
    best_model: str
    model_scores: dict[str, float]
    all_predictions: dict[str, pl.DataFrame] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)


# Map plan candidate names to forecast functions
_MODEL_REGISTRY: dict[str, str] = {
    "naive": "polars_ts.models.baselines.naive_forecast",
    "moving_average": "polars_ts.models.baselines.moving_average_forecast",
    "ses": "polars_ts.models.exponential_smoothing.ses_forecast",
    "holt": "polars_ts.models.exponential_smoothing.holt_forecast",
    "holt_winters": "polars_ts.models.exponential_smoothing.holt_winters_forecast",
}


def _import_model(dotted_path: str) -> Any:
    import importlib

    module_path, _, func_name = dotted_path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


class ForecasterAgent:
    """Fits candidate models, validates, and selects the best or builds an ensemble.

    Parameters
    ----------
    backend
        LLM backend for guided decisions. Defaults to rule-based heuristics.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.

    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def forecast(self, df: pl.DataFrame, plan: ForecastPlan) -> ForecastAgentResult:
        """Fit each candidate model and select the best by validation MAE."""
        h = plan.horizon

        # Split into train / validation
        train, val = self._train_val_split(df, h)

        scores: dict[str, float] = {}
        all_preds: dict[str, pl.DataFrame] = {}

        for name in plan.candidates:
            dotted = _MODEL_REGISTRY.get(name)
            if dotted is None:
                continue
            try:
                fn = _import_model(dotted)
                extra = plan.config.get(name, {})
                preds = fn(train, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra)
                all_preds[name] = preds
                mae = self._score(val, preds)
                scores[name] = mae
            except Exception:
                scores[name] = float("inf")

        if not scores:
            # Fallback: just use naive
            from polars_ts.models.baselines import naive_forecast

            preds = naive_forecast(train, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col)
            return ForecastAgentResult(
                predictions=preds,
                best_model="naive",
                model_scores={"naive": float("nan")},
                all_predictions={"naive": preds},
            )

        best_name = min(scores, key=lambda k: scores[k])

        # Build ensemble if requested and we have 2+ valid models
        valid_models = {k: v for k, v in scores.items() if np.isfinite(v) and v > 0}
        if plan.ensemble and len(valid_models) >= 2:
            return self._ensemble_forecast(df, plan, scores, all_preds, valid_models, best_name)

        # Single best model: re-fit on full data
        best_fn = _import_model(_MODEL_REGISTRY[best_name])
        extra = plan.config.get(best_name, {})
        final_preds = best_fn(df, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra)

        return ForecastAgentResult(
            predictions=final_preds,
            best_model=best_name,
            model_scores=scores,
            all_predictions=all_preds,
        )

    def _ensemble_forecast(
        self,
        df: pl.DataFrame,
        plan: ForecastPlan,
        scores: dict[str, float],
        all_preds: dict[str, pl.DataFrame],
        valid_models: dict[str, float],
        best_name: str,
    ) -> ForecastAgentResult:
        """Build a weighted ensemble of top models using inverse-MAE weights."""
        h = plan.horizon

        # Compute inverse-MAE weights
        inv_mae = {k: 1.0 / v for k, v in valid_models.items()}
        total = sum(inv_mae.values())
        weights = {k: v / total for k, v in inv_mae.items()}

        # Re-fit each ensemble member on full data and combine
        ensemble_preds_list: list[tuple[str, float, pl.DataFrame]] = []
        for name, w in weights.items():
            dotted = _MODEL_REGISTRY.get(name)
            if dotted is None:
                continue
            try:
                fn = _import_model(dotted)
                extra = plan.config.get(name, {})
                preds = fn(df, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra)
                ensemble_preds_list.append((name, w, preds))
            except Exception:
                continue

        if not ensemble_preds_list:
            # Fallback to best single model
            best_fn = _import_model(_MODEL_REGISTRY[best_name])
            extra = plan.config.get(best_name, {})
            final_preds = best_fn(
                df, h=h, target_col=self.target_col, id_col=self.id_col, time_col=self.time_col, **extra
            )
            return ForecastAgentResult(
                predictions=final_preds,
                best_model=best_name,
                model_scores=scores,
                all_predictions=all_preds,
            )

        # Weighted average of predictions
        final_preds = self._weighted_average(ensemble_preds_list)

        # Normalize weights for the models that actually contributed
        contrib_total = sum(w for _, w, _ in ensemble_preds_list)
        final_weights = {name: w / contrib_total for name, w, _ in ensemble_preds_list}

        return ForecastAgentResult(
            predictions=final_preds,
            best_model=f"ensemble({', '.join(final_weights.keys())})",
            model_scores=scores,
            all_predictions=all_preds,
            ensemble_weights=final_weights,
        )

    def _weighted_average(
        self,
        preds_list: list[tuple[str, float, pl.DataFrame]],
    ) -> pl.DataFrame:
        """Compute weighted average of y_hat across model predictions."""
        # Use the first prediction's structure as the base
        _, first_w, base = preds_list[0]
        join_cols = [c for c in [self.id_col, self.time_col] if c in base.columns]

        result = base.with_columns((pl.col("y_hat") * first_w).alias("y_hat_weighted"))

        for _, w, preds in preds_list[1:]:
            right = preds.select([*join_cols, (pl.col("y_hat") * w).alias("__y_hat_part")])
            result = result.join(right, on=join_cols, how="left")
            result = result.with_columns(
                (pl.col("y_hat_weighted") + pl.col("__y_hat_part").fill_null(0)).alias("y_hat_weighted")
            )
            result = result.drop("__y_hat_part")

        # Replace y_hat with weighted sum
        result = result.with_columns(pl.col("y_hat_weighted").alias("y_hat")).drop("y_hat_weighted")
        return result

    def _train_val_split(
        self,
        df: pl.DataFrame,
        h: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split each series: last h observations go to validation."""
        has_id = self.id_col in df.columns
        sort_cols = [self.id_col, self.time_col] if has_id else [self.time_col]
        sorted_df = df.sort(*sort_cols)
        if has_id:
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).over(self.id_col).alias("__rank")
            )
        else:
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).alias("__rank")
            )
        train = ranked.filter(pl.col("__rank") > h).drop("__rank")
        val = ranked.filter(pl.col("__rank") <= h).drop("__rank")
        return train, val

    def _score(self, val: pl.DataFrame, preds: pl.DataFrame) -> float:
        """Compute MAE between validation actuals and predictions."""
        join_cols = [c for c in [self.id_col, self.time_col] if c in val.columns and c in preds.columns]
        if not join_cols:
            return float("inf")

        joined = val.join(preds, on=join_cols, how="inner")
        if joined.is_empty():
            return float("inf")

        actual = joined[self.target_col].to_numpy().astype(np.float64)
        predicted = joined["y_hat"].to_numpy().astype(np.float64)
        return _compute_mae(actual, predicted)
