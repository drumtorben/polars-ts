"""Weighted forecast ensemble.

Combines multiple forecast DataFrames using equal, manual, or
inverse-error-optimized weights.
"""

from __future__ import annotations

import polars as pl


def _join_forecasts(
    forecasts: list[pl.DataFrame],
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Join multiple forecast DataFrames on (id_col, time_col).

    Renames each ``y_hat`` column to ``y_hat_0``, ``y_hat_1``, etc.
    Validates that all forecasts share the same (id_col, time_col) rows.

    Returns
    -------
    pl.DataFrame
        Joined DataFrame with columns ``[id_col, time_col, y_hat_0, y_hat_1, ...]``.

    """
    if not forecasts:
        raise ValueError("forecasts must be a non-empty list")

    join_cols = [id_col, time_col] if id_col in forecasts[0].columns else [time_col]

    base = forecasts[0].select(*join_cols, pl.col("y_hat").alias("y_hat_0"))
    n_rows = len(base)

    for i, fc in enumerate(forecasts[1:], start=1):
        renamed = fc.select(*join_cols, pl.col("y_hat").alias(f"y_hat_{i}"))
        base = base.join(renamed, on=join_cols, how="inner")
        if len(base) != n_rows:
            raise ValueError(
                f"Forecast {i} has different (id, time) rows than forecast 0 "
                f"({len(renamed)} vs {n_rows} rows; {len(base)} matched)"
            )

    return base


class WeightedEnsemble:
    """Combine multiple forecasts using weighted averaging.

    Accepts pre-computed forecast DataFrames and combines them.
    No model training is performed.

    Parameters
    ----------
    weights
        Weighting strategy:

        - ``"equal"`` (default): all models get equal weight ``1/n``.
        - ``"inverse_error"``: weights are ``1/MAE`` normalized to sum
          to 1. Requires ``validation_pairs`` in :meth:`combine`.
        - A ``list[float]``: explicit weights (will be normalized).
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.

    """

    def __init__(
        self,
        weights: str | list[float] = "equal",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        self.weights = weights
        self.id_col = id_col
        self.time_col = time_col

    def combine(
        self,
        forecasts: list[pl.DataFrame],
        validation_dfs: list[pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Combine forecast DataFrames into a single ensemble forecast.

        Parameters
        ----------
        forecasts
            List of forecast DataFrames, each with ``[id_col, time_col, "y_hat"]``.
        validation_dfs
            Required when ``weights="inverse_error"``. List of
            DataFrames (one per model), each with ``y`` and ``y_hat``
            columns for computing per-model MAE.

        Returns
        -------
        pl.DataFrame
            Combined forecast with columns ``[id_col, time_col, "y_hat"]``.

        """
        if not forecasts:
            raise ValueError("forecasts must be a non-empty list")
        if len(forecasts) < 2:
            raise ValueError("Need at least 2 forecasts to ensemble")

        n = len(forecasts)
        w = self._resolve_weights(n, validation_dfs)

        joined = _join_forecasts(forecasts, self.id_col, self.time_col)
        hat_cols = [f"y_hat_{i}" for i in range(n)]

        # Weighted sum
        expr = pl.lit(0.0)
        for i, col_name in enumerate(hat_cols):
            expr = expr + pl.col(col_name) * w[i]

        join_cols = [c for c in [self.id_col, self.time_col] if c in joined.columns]
        return joined.select(*join_cols, expr.alias("y_hat"))

    def _resolve_weights(
        self,
        n: int,
        validation_dfs: list[pl.DataFrame] | None,
    ) -> list[float]:
        if isinstance(self.weights, list):
            if len(self.weights) != n:
                raise ValueError(f"Expected {n} weights, got {len(self.weights)}")
            total = sum(self.weights)
            if total <= 0:
                raise ValueError("Weights must sum to a positive value")
            return [w / total for w in self.weights]

        if self.weights == "equal":
            return [1.0 / n] * n

        if self.weights == "inverse_error":
            if validation_dfs is None:
                raise ValueError("validation_dfs required when weights='inverse_error'")
            if len(validation_dfs) != n:
                raise ValueError(f"Expected {n} validation DataFrames, got {len(validation_dfs)}")
            return self._compute_inverse_error_weights(validation_dfs)

        raise ValueError(f"Unknown weights strategy: {self.weights!r}")

    @staticmethod
    def _compute_inverse_error_weights(
        validation_dfs: list[pl.DataFrame],
    ) -> list[float]:
        from polars_ts.metrics.forecast import mae

        errors: list[float] = []
        for df in validation_dfs:
            if "y" not in df.columns or "y_hat" not in df.columns:
                raise ValueError("Validation DataFrames must have 'y' and 'y_hat' columns")
            score = mae(df, actual_col="y", predicted_col="y_hat")
            if not isinstance(score, float):
                score = score.item()  # type: ignore[union-attr]
            if score <= 0:
                score = 1e-10  # Avoid division by zero for perfect models
            errors.append(score)

        inv = [1.0 / e for e in errors]
        total = sum(inv)
        return [w / total for w in inv]
