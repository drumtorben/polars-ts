"""Reporting mixin for CausalImpact — summary, to_frame, placebo_test."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from polars_ts.causal.causal_impact_results import CausalImpactResult


class CausalImpactReportingMixin:
    """Mixin providing summary, to_frame, and placebo_test methods.

    Attributes below are declared for mypy — they are set by the
    consuming ``CausalImpact`` class.
    """

    # Declared for type checking — set by CausalImpact.__init__
    is_fitted_: bool
    _states: dict[Any, Any]
    _intervention_date: date | datetime | None
    id_col: str
    time_col: str
    target_col: str
    trend: str
    seasonal: int | None
    sigma_obs: float
    sigma_level: float
    sigma_trend: float
    sigma_seasonal: float
    coverage: float
    covariates: list[str]
    covariate_role: dict[str, Literal["pre_only", "always"]] | None

    def results(self) -> dict[Any, CausalImpactResult]:
        """Return per-series CausalImpactResult objects.

        Returns
        -------
        dict[Any, CausalImpactResult]
            Mapping from series ID to result.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before results().")
        return {gid: s.result for gid, s in self._states.items() if s.result is not None}

    def summary(self) -> pl.DataFrame:
        """Return a summary DataFrame with one row per series.

        Columns: id_col, total_effect, total_effect_lower, total_effect_upper,
        relative_effect, relative_effect_lower, relative_effect_upper,
        pre_mape, pre_coverage.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before summary().")

        rows: list[dict[str, Any]] = []
        for gid, state in self._states.items():
            r = state.result
            assert r is not None
            rows.append(
                {
                    self.id_col: gid,
                    "total_effect": r.total_effect,
                    "total_effect_lower": r.total_effect_lower,
                    "total_effect_upper": r.total_effect_upper,
                    "relative_effect": r.relative_effect,
                    "relative_effect_lower": r.relative_effect_lower,
                    "relative_effect_upper": r.relative_effect_upper,
                    "pre_mape": r.pre_mape,
                    "pre_coverage": r.pre_coverage,
                }
            )
        return pl.DataFrame(rows)

    def to_frame(self) -> pl.DataFrame:
        """Return pointwise results as a DataFrame.

        Columns: id_col, step, observed, counterfactual, counterfactual_lower,
        counterfactual_upper, point_effect, point_effect_lower,
        point_effect_upper, cumulative_effect, cumulative_effect_lower,
        cumulative_effect_upper.

        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before to_frame().")

        all_rows: list[dict[str, Any]] = []
        for gid, state in self._states.items():
            r = state.result
            assert r is not None
            for t in range(state.post_len):
                all_rows.append(
                    {
                        self.id_col: gid,
                        "step": t + 1,
                        "observed": float(r.observed_post[t]),
                        "counterfactual": float(r.counterfactual[t]),
                        "counterfactual_lower": float(r.counterfactual_lower[t]),
                        "counterfactual_upper": float(r.counterfactual_upper[t]),
                        "point_effect": float(r.point_effect[t]),
                        "point_effect_lower": float(r.point_effect_lower[t]),
                        "point_effect_upper": float(r.point_effect_upper[t]),
                        "cumulative_effect": float(r.cumulative_effect[t]),
                        "cumulative_effect_lower": float(r.cumulative_effect_lower[t]),
                        "cumulative_effect_upper": float(r.cumulative_effect_upper[t]),
                    }
                )
        return pl.DataFrame(all_rows)

    def placebo_test(
        self,
        df: pl.DataFrame,
        placebo_date: date | datetime,
    ) -> pl.DataFrame:
        """Run a placebo test at a date before the actual intervention.

        Fits the model pretending ``placebo_date`` is the intervention,
        using only data from the pre-intervention period (data after
        the real intervention is excluded to avoid contamination).
        If the model is well-specified, the estimated effect should
        be near zero.

        Parameters
        ----------
        df
            Same panel DataFrame used in ``fit()``.
        placebo_date
            A date strictly before the actual intervention.

        Returns
        -------
        pl.DataFrame
            Summary with columns: id_col, total_effect, total_effect_lower,
            total_effect_upper, relative_effect.

        """
        if not self.is_fitted_ or self._intervention_date is None:
            raise RuntimeError("Call fit() before placebo_test().")

        # Import here to avoid circular import at module level
        from polars_ts.causal.causal_impact import CausalImpact

        # Filter out post-intervention data to avoid contamination
        pre_only = df.filter(pl.col(self.time_col) < self._intervention_date)

        placebo = CausalImpact(
            trend=self.trend,
            seasonal=self.seasonal,
            sigma_obs=self.sigma_obs,
            sigma_level=self.sigma_level,
            sigma_trend=self.sigma_trend,
            sigma_seasonal=self.sigma_seasonal,
            coverage=self.coverage,
            covariates=self.covariates if self.covariates else None,
            covariate_role=self.covariate_role,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        placebo.fit(pre_only, intervention_date=placebo_date)
        return placebo.summary()
