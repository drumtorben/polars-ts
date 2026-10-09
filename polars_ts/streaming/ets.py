"""Online exponential smoothing (SES, Holt, Holt-Winters) with partial_fit."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from polars_ts.models.baselines import _infer_freq, _make_future_dates


class StreamingETS:
    """Streaming exponential smoothing with incremental updates.

    Parameters
    ----------
    method
        One of ``"ses"``, ``"holt"``, ``"holt_winters"``.
    alpha
        Level smoothing parameter.
    beta
        Trend smoothing parameter (Holt / Holt-Winters only).
    gamma
        Seasonal smoothing parameter (Holt-Winters only).
    season_length
        Observations per season (Holt-Winters only).
    seasonal
        ``"additive"`` or ``"multiplicative"`` (Holt-Winters only).
    id_col, time_col, target_col
        Column name overrides.

    """

    def __init__(
        self,
        method: str = "ses",
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        season_length: int = 7,
        seasonal: str = "additive",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if method in ("holt", "holt_winters") and not (0 < beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if method == "holt_winters" and not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if method == "holt_winters" and season_length < 2:
            raise ValueError(f"season_length must be >= 2, got {season_length}")
        if method not in ("ses", "holt", "holt_winters"):
            raise ValueError(f"method must be 'ses', 'holt', or 'holt_winters', got {method!r}")

        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.seasonal = seasonal
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self.is_fitted_ = False
        self._state: dict[Any, dict[str, Any]] = {}
        self._freq: timedelta | int | float | None = None
        self._schema: dict[str, Any] = {}

    @property
    def state_(self) -> dict[Any, dict[str, Any]]:
        """Per-series state dictionaries."""
        return self._state

    def fit(self, df: pl.DataFrame) -> StreamingETS:
        """Initialize state from a batch of historical data."""
        sorted_df = df.sort(self.id_col, self.time_col)
        self._freq = _infer_freq(sorted_df[self.time_col])
        self._schema = {
            self.id_col: df.schema[self.id_col],
            self.time_col: df.schema[self.time_col],
        }

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            sid = group_id[0]
            values = [float(v) for v in group_df[self.target_col].to_list()]
            last_time = group_df[self.time_col][-1]
            self._state[sid] = self._fit_series(values, last_time)

        self.is_fitted_ = True
        return self

    def partial_fit(self, df: pl.DataFrame) -> StreamingETS:
        """Incrementally update state with new observations."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before partial_fit()")

        sorted_df = df.sort(self.id_col, self.time_col)
        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            sid = group_id[0]
            values = [float(v) for v in group_df[self.target_col].to_list()]
            last_time = group_df[self.time_col][-1]

            if sid not in self._state:
                # New series — initialize from scratch
                self._state[sid] = self._fit_series(values, last_time)
            else:
                self._update_series(sid, values, last_time)

        return self

    def predict(self, h: int) -> pl.DataFrame:
        """Generate h-step-ahead forecasts from current state."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")

        assert self._freq is not None
        rows: list[dict[str, Any]] = []
        for sid, state in self._state.items():
            forecasts = self._forecast_series(state, h)
            future_times = _make_future_dates(state["last_time"], self._freq, h)
            for t, fc in zip(future_times, forecasts, strict=False):
                rows.append({self.id_col: sid, self.time_col: t, "y_hat": fc})

        schema = {**self._schema, "y_hat": pl.Float64()}
        return pl.DataFrame(rows, schema=schema).sort(self.id_col, self.time_col)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_series(self, values: list[float], last_time: Any) -> dict[str, Any]:
        """Compute initial state for one series."""
        if self.method == "ses":
            level = values[0]
            for v in values[1:]:
                level = self.alpha * v + (1 - self.alpha) * level
            return {"level": level, "last_time": last_time, "n_obs": len(values)}

        elif self.method == "holt":
            level = values[0]
            trend = values[1] - values[0] if len(values) > 1 else 0.0
            for v in values[1:]:
                prev_level = level
                level = self.alpha * v + (1 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            return {
                "level": level,
                "trend": trend,
                "last_time": last_time,
                "n_obs": len(values),
            }

        elif self.method == "holt_winters":
            m = self.season_length
            if len(values) < 2 * m:
                raise ValueError(
                    f"Holt-Winters requires at least 2*season_length={2 * m} observations, got {len(values)}"
                )
            additive = self.seasonal == "additive"
            first_season_avg = sum(values[:m]) / m
            level = first_season_avg
            trend = (sum(values[m : 2 * m]) / m - first_season_avg) / m

            if additive:
                seasons = [values[i] - first_season_avg for i in range(m)]
            else:
                seasons = [values[i] / first_season_avg if first_season_avg != 0 else 1.0 for i in range(m)]

            n = len(values)
            for t in range(m, n):
                v = values[t]
                s_idx = t % m
                prev_level = level
                if additive:
                    level = self.alpha * (v - seasons[s_idx]) + (1 - self.alpha) * (level + trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
                    seasons[s_idx] = self.gamma * (v - level) + (1 - self.gamma) * seasons[s_idx]
                else:
                    denom = seasons[s_idx] if seasons[s_idx] != 0 else 1.0
                    level = self.alpha * (v / denom) + (1 - self.alpha) * (level + trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
                    denom_l = level if level != 0 else 1.0
                    seasons[s_idx] = self.gamma * (v / denom_l) + (1 - self.gamma) * seasons[s_idx]

            return {
                "level": level,
                "trend": trend,
                "seasons": seasons,
                "season_idx": (n - 1) % m,
                "last_time": last_time,
                "n_obs": n,
            }
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

    def _update_series(self, sid: Any, values: list[float], last_time: Any) -> None:
        """Incrementally update state for one series."""
        state = self._state[sid]

        if self.method == "ses":
            level = state["level"]
            for v in values:
                level = self.alpha * v + (1 - self.alpha) * level
            state["level"] = level
            state["last_time"] = last_time
            state["n_obs"] += len(values)

        elif self.method == "holt":
            level = state["level"]
            trend = state["trend"]
            for v in values:
                prev_level = level
                level = self.alpha * v + (1 - self.alpha) * (level + trend)
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            state["level"] = level
            state["trend"] = trend
            state["last_time"] = last_time
            state["n_obs"] += len(values)

        elif self.method == "holt_winters":
            m = self.season_length
            additive = self.seasonal == "additive"
            level = state["level"]
            trend = state["trend"]
            seasons = state["seasons"]
            s_offset = state["season_idx"] + 1

            for i, v in enumerate(values):
                s_idx = (s_offset + i) % m
                prev_level = level
                if additive:
                    level = self.alpha * (v - seasons[s_idx]) + (1 - self.alpha) * (level + trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
                    seasons[s_idx] = self.gamma * (v - level) + (1 - self.gamma) * seasons[s_idx]
                else:
                    denom = seasons[s_idx] if seasons[s_idx] != 0 else 1.0
                    level = self.alpha * (v / denom) + (1 - self.alpha) * (level + trend)
                    trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
                    denom_l = level if level != 0 else 1.0
                    seasons[s_idx] = self.gamma * (v / denom_l) + (1 - self.gamma) * seasons[s_idx]

            state["level"] = level
            state["trend"] = trend
            state["seasons"] = seasons
            state["season_idx"] = (s_offset + len(values) - 1) % m
            state["last_time"] = last_time
            state["n_obs"] += len(values)

    def _forecast_series(self, state: dict[str, Any], h: int) -> list[float]:
        """Generate forecasts from current state."""
        if self.method == "ses":
            return [state["level"]] * h

        elif self.method == "holt":
            level = state["level"]
            trend = state["trend"]
            return [level + step * trend for step in range(1, h + 1)]

        elif self.method == "holt_winters":
            m = self.season_length
            additive = self.seasonal == "additive"
            level = state["level"]
            trend = state["trend"]
            seasons = state["seasons"]
            s_offset = state["season_idx"]

            forecasts = []
            for step in range(1, h + 1):
                s_idx = (s_offset + step) % m
                if additive:
                    forecasts.append(level + step * trend + seasons[s_idx])
                else:
                    forecasts.append((level + step * trend) * seasons[s_idx])
            return forecasts

        raise ValueError(f"Unknown method: {self.method!r}")
