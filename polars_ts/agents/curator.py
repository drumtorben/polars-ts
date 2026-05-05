"""CuratorAgent: LLM-guided data diagnostics and targeted preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_ts.agents._protocol import LLMBackend, RuleBasedBackend


@dataclass
class CurationReport:
    """Results of data diagnostics."""

    n_observations: int
    n_series: int
    n_missing: int
    n_outliers: int
    detected_period: int | None
    has_trend: bool
    is_stationary: bool
    recommended_lookback: int | None
    summary: str


class CuratorAgent:
    """Diagnoses data quality issues and applies targeted preprocessing.

    Parameters
    ----------
    backend
        LLM backend for guided diagnostics. Defaults to rule-based heuristics.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with target values.
    outlier_threshold
        Z-score threshold for outlier detection.

    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        outlier_threshold: float = 3.0,
    ) -> None:
        self.backend = backend or RuleBasedBackend()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.outlier_threshold = outlier_threshold

    def curate(self, df: pl.DataFrame) -> CurationReport:
        """Run diagnostics on the input data and return a report."""
        n_obs = len(df)
        n_missing = int(df[self.target_col].null_count() + df[self.target_col].is_nan().sum())

        if self.id_col in df.columns:
            n_series = df[self.id_col].n_unique()
        else:
            n_series = 1

        n_outliers = self._count_outliers(df)
        detected_period = self._detect_period(df)
        has_trend = self._detect_trend(df)
        is_stationary = self._check_stationarity(df)
        recommended_lookback = self._recommend_lookback(df)

        summary = f"{n_series} series, {n_obs} obs, {n_missing} missing, {n_outliers} outliers"
        if detected_period:
            summary += f", period={detected_period}"
        if has_trend:
            summary += ", trend detected"
        if not is_stationary:
            summary += ", non-stationary"
        if recommended_lookback:
            summary += f", lookback={recommended_lookback}"

        # Enhance summary with LLM if available
        if not isinstance(self.backend, RuleBasedBackend):
            llm_summary = self.backend.complete(f"Summarize these time series diagnostics concisely:\n{summary}")
            if llm_summary:
                summary = llm_summary

        return CurationReport(
            n_observations=n_obs,
            n_series=n_series,
            n_missing=n_missing,
            n_outliers=n_outliers,
            detected_period=detected_period,
            has_trend=has_trend,
            is_stationary=is_stationary,
            recommended_lookback=recommended_lookback,
            summary=summary,
        )

    def curate_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run diagnostics then apply imputation and outlier treatment."""
        result = df

        # Impute missing values with forward fill then backward fill
        if self.id_col in result.columns:
            result = result.sort(self.id_col, self.time_col)
            result = result.with_columns(
                pl.col(self.target_col).fill_nan(None).forward_fill().over(self.id_col).alias(self.target_col)
            )
            result = result.with_columns(
                pl.col(self.target_col).backward_fill().over(self.id_col).alias(self.target_col)
            )
        else:
            result = result.sort(self.time_col)
            result = result.with_columns(pl.col(self.target_col).fill_nan(None).forward_fill().alias(self.target_col))
            result = result.with_columns(pl.col(self.target_col).backward_fill().alias(self.target_col))

        # Clip outliers to +-threshold * std per series
        result = self._clip_outliers(result)

        return result

    def trim_lookback(self, df: pl.DataFrame, lookback: int | None = None) -> pl.DataFrame:
        """Trim each series to the most recent ``lookback`` observations.

        Parameters
        ----------
        df
            Input DataFrame.
        lookback
            Number of most-recent observations to keep per series.
            If *None*, uses the recommended lookback from diagnostics.

        """
        if lookback is None:
            lookback = self._recommend_lookback(df)
        if lookback is None:
            return df

        if self.id_col in df.columns:
            sorted_df = df.sort(self.id_col, self.time_col)
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).over(self.id_col).alias("__rank")
            )
        else:
            sorted_df = df.sort(self.time_col)
            ranked = sorted_df.with_columns(
                pl.col(self.time_col).rank(method="ordinal", descending=True).alias("__rank")
            )

        return ranked.filter(pl.col("__rank") <= lookback).drop("__rank")

    def _count_outliers(self, df: pl.DataFrame) -> int:
        total = 0
        if self.id_col in df.columns:
            for _, group_df in df.group_by(self.id_col, maintain_order=True):
                total += self._count_outliers_single(group_df)
        else:
            total = self._count_outliers_single(df)
        return total

    def _count_outliers_single(self, df: pl.DataFrame) -> int:
        values = df[self.target_col].drop_nulls().drop_nans().to_numpy()
        if len(values) < 3:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        z = np.abs((values - mean) / std)
        return int(np.sum(z > self.outlier_threshold))

    def _detect_period(self, df: pl.DataFrame) -> int | None:
        """Detect dominant period via autocorrelation peak."""
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        if len(values) < 10:
            return None

        values = values - np.mean(values)
        n = len(values)
        var = np.dot(values, values)
        if var == 0:
            return None

        max_lag = min(n // 2, 60)
        best_lag = None
        best_acf = 0.0
        for lag in range(2, max_lag):
            acf_val = np.dot(values[: n - lag], values[lag:]) / var
            if acf_val > best_acf:
                best_acf = acf_val
                best_lag = lag

        if best_acf > 0.3:
            return best_lag
        return None

    def _detect_trend(self, df: pl.DataFrame) -> bool:
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        if len(values) < 5:
            return False

        x = np.arange(len(values), dtype=np.float64)
        slope = np.polyfit(x, values, 1)[0]
        value_range = np.ptp(values)
        if value_range == 0:
            return False
        return bool(abs(slope * len(values)) / value_range > 0.2)

    def _check_stationarity(self, df: pl.DataFrame) -> bool:
        """Check stationarity by comparing mean/variance of first vs second half."""
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        if len(values) < 20:
            return True

        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        mean_ratio = abs(np.mean(first_half) - np.mean(second_half)) / (np.std(values) + 1e-10)
        var_ratio = np.std(second_half) / (np.std(first_half) + 1e-10)

        # Non-stationary if mean shifted significantly or variance ratio far from 1
        return bool(mean_ratio < 1.5 and 0.5 < var_ratio < 2.0)

    def _recommend_lookback(self, df: pl.DataFrame) -> int | None:
        """Recommend lookback window based on regime change detection.

        Uses a rolling variance ratio to detect the most recent structural break,
        then recommends using only data from after the break.
        """
        if self.id_col in df.columns:
            first_id = df[self.id_col][0]
            values = df.filter(pl.col(self.id_col) == first_id)[self.target_col].drop_nulls().drop_nans().to_numpy()
        else:
            values = df[self.target_col].drop_nulls().drop_nans().to_numpy()

        n = len(values)
        if n < 40:
            return None

        window = max(n // 10, 10)
        # Compute rolling variance ratio between adjacent windows
        best_break = None
        best_score = 0.0

        for i in range(window, n - window):
            left_var = np.var(values[i - window : i])
            right_var = np.var(values[i : i + window])
            if left_var < 1e-10 and right_var < 1e-10:
                continue
            ratio = max(left_var, right_var) / (min(left_var, right_var) + 1e-10)
            # Also check mean shift
            left_mean = np.mean(values[i - window : i])
            right_mean = np.mean(values[i : i + window])
            mean_shift = abs(left_mean - right_mean) / (np.std(values) + 1e-10)
            score = ratio + mean_shift

            if score > best_score:
                best_score = score
                best_break = i

        # Only recommend trimming if there's a substantial regime change
        if best_break is not None and best_score > 4.0:
            return n - best_break

        return None

    def _clip_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.id_col in df.columns:
            mean_expr = pl.col(self.target_col).mean().over(self.id_col)
            std_expr = pl.col(self.target_col).std().over(self.id_col)
        else:
            mean_expr = pl.col(self.target_col).mean()
            std_expr = pl.col(self.target_col).std()

        lower = mean_expr - self.outlier_threshold * std_expr
        upper = mean_expr + self.outlier_threshold * std_expr

        return df.with_columns(pl.col(self.target_col).clip(lower, upper).alias(self.target_col))
