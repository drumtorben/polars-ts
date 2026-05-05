"""CausalImpact: Bayesian causal inference for intervention analysis.

Estimates the causal effect of an intervention on a time series using
a Bayesian structural time series (BSTS) counterfactual model. The
pre-intervention period trains the model; the post-intervention
counterfactual projection is subtracted from the observed series to
yield the estimated treatment effect with credible intervals.

Design notes (from issue #148 feedback):
- Returns (point, lower, upper) from day one — no bolt-on bootstrap.
- Exposes full BSTS spec so priors are never hidden.
- Pre-period diagnostics run by default.
- Built-in placebo tests via ``placebo_test``.
- Covariates with ``covariate_role`` guard to prevent post-treatment
  bias (issue #185).

References
----------
Brodersen et al. (2015). *Inferring causal impact using Bayesian
structural time series models.* Annals of Applied Statistics.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import polars as pl

from polars_ts.bayesian.bsts import BSTS, BSTSResult


@dataclass
class CausalImpactResult:
    """Result container for a CausalImpact analysis.

    All effect arrays have length equal to the post-intervention period.

    Attributes
    ----------
    point_effect
        Pointwise causal effect (observed - counterfactual).
    point_effect_lower
        Lower credible bound of the pointwise effect.
    point_effect_upper
        Upper credible bound of the pointwise effect.
    cumulative_effect
        Cumulative sum of pointwise effects.
    cumulative_effect_lower
        Conservative lower bound of cumulative effect (sum of
        pointwise lower bounds). This is wider than a proper
        posterior-based interval because it does not account for
        cross-timestep correlation.
    cumulative_effect_upper
        Conservative upper bound of cumulative effect.
    total_effect
        Sum of pointwise effects over the post period.
    total_effect_lower
        Lower credible bound of total effect.
    total_effect_upper
        Upper credible bound of total effect.
    relative_effect
        Total effect divided by sum of counterfactual.
    relative_effect_lower
        Lower credible bound of relative effect.
    relative_effect_upper
        Upper credible bound of relative effect.
    counterfactual
        Predicted counterfactual series for the post period.
    counterfactual_lower
        Lower credible bound of counterfactual.
    counterfactual_upper
        Upper credible bound of counterfactual.
    observed_post
        Observed values in the post period.
    bsts_result
        Underlying BSTS model result.
    pre_mape
        Mean absolute percentage error on the pre-period (diagnostic).
    pre_coverage
        Fraction of pre-period observations inside the credible interval.

    """

    point_effect: np.ndarray
    point_effect_lower: np.ndarray
    point_effect_upper: np.ndarray
    cumulative_effect: np.ndarray
    cumulative_effect_lower: np.ndarray
    cumulative_effect_upper: np.ndarray
    total_effect: float
    total_effect_lower: float
    total_effect_upper: float
    relative_effect: float
    relative_effect_lower: float
    relative_effect_upper: float
    counterfactual: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    observed_post: np.ndarray
    bsts_result: BSTSResult
    pre_mape: float
    pre_coverage: float


@dataclass
class _FitState:
    """Internal per-series fit state."""

    bsts_model: BSTS
    pre_y: np.ndarray
    post_y: np.ndarray
    pre_len: int
    post_len: int
    result: CausalImpactResult | None = None


def _validate_covariate_roles(
    covariates: list[str],
    covariate_role: dict[str, Literal["pre_only", "always"]] | None,
) -> dict[str, Literal["pre_only", "always"]]:
    """Validate and return resolved covariate roles.

    Raises warnings for covariates without explicit role assignment.
    """
    if covariate_role is None:
        if covariates:
            warnings.warn(
                "Covariates passed without explicit covariate_role. "
                "All covariates default to 'always' and will be used in "
                "post-period counterfactual projection. If any are "
                "post-treatment variables, this will bias the estimate "
                "toward zero. Set covariate_role explicitly.",
                UserWarning,
                stacklevel=3,
            )
        return {c: "always" for c in covariates}

    resolved: dict[str, Literal["pre_only", "always"]] = {}
    for c in covariates:
        if c not in covariate_role:
            warnings.warn(
                f"Covariate {c!r} has no explicit role in covariate_role. " f"Defaulting to 'always'.",
                UserWarning,
                stacklevel=3,
            )
            resolved[c] = "always"
        else:
            resolved[c] = covariate_role[c]

    if all(r == "always" for r in resolved.values()) and len(resolved) > 0:
        warnings.warn(
            "All covariates are marked 'always'. Verify that none are "
            "post-treatment variables (they would bias the estimate "
            "toward zero).",
            UserWarning,
            stacklevel=3,
        )

    return resolved


def _fit_regression(
    pre_y: np.ndarray,
    pre_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit demeaned OLS regression on pre-period data.

    Demeaning avoids the regression absorbing the level/trend that
    the BSTS model should capture. Returns ``(beta, X_mean, y_mean)``
    where beta is shape ``(n_covariates,)``.
    """
    y_mean = float(np.mean(pre_y))
    X_mean = np.mean(pre_X, axis=0)
    beta, _, _, _ = np.linalg.lstsq(pre_X - X_mean, pre_y - y_mean, rcond=None)
    return beta, X_mean, y_mean


class CausalImpact:
    """Bayesian CausalImpact estimator.

    Parameters
    ----------
    trend
        BSTS trend type: ``"level"`` or ``"local_linear"``.
    seasonal
        Number of seasons for the BSTS seasonal component.
        ``None`` disables seasonality.
    sigma_obs
        Observation noise standard deviation.
    sigma_level
        Level component noise standard deviation.
    sigma_trend
        Trend component noise standard deviation.
    sigma_seasonal
        Seasonal component noise standard deviation.
    coverage
        Credible interval coverage (e.g. 0.9 for 90%).
    covariates
        Column names of exogenous covariates. When provided, an OLS
        regression is fit on the pre-period, and the regression
        component is added to the BSTS counterfactual.
    covariate_role
        Mapping from covariate name to role:

        - ``"always"``: covariate is used in both pre- and post-period.
        - ``"pre_only"``: covariate is used only for pre-period fitting
          and excluded from post-period counterfactual projection
          (prevents post-treatment bias).

        Covariates not listed default to ``"always"`` with a warning.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    target_col
        Column with observed values.

    Notes
    -----
    The BSTS prior hyperparameters (``sigma_*``) are exposed explicitly
    because the posterior interval is dominated by the prior when the
    pre-period is short (<60 observations). Always inspect ``pre_mape``
    and ``pre_coverage`` diagnostics before trusting effect estimates.

    Post-treatment covariates must be excluded from the counterfactual
    because they encode the treatment effect as a nuisance variable,
    biasing the estimate toward zero. Use ``covariate_role="pre_only"``
    for any variable that may be affected by the intervention.

    """

    def __init__(
        self,
        trend: str = "local_linear",
        seasonal: int | None = None,
        sigma_obs: float = 1.0,
        sigma_level: float = 0.1,
        sigma_trend: float = 0.01,
        sigma_seasonal: float = 0.01,
        coverage: float = 0.9,
        covariates: list[str] | None = None,
        covariate_role: dict[str, Literal["pre_only", "always"]] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.sigma_obs = sigma_obs
        self.sigma_level = sigma_level
        self.sigma_trend = sigma_trend
        self.sigma_seasonal = sigma_seasonal
        self.coverage = coverage
        self.covariates = covariates or []
        self.covariate_role = covariate_role
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._resolved_roles: dict[str, Literal["pre_only", "always"]] = {}
        self._states: dict[Any, _FitState] = {}
        self._intervention_date: date | datetime | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pl.DataFrame,
        intervention_date: date | datetime,
    ) -> CausalImpact:
        """Fit the causal impact model.

        Parameters
        ----------
        df
            Panel DataFrame with ``id_col``, ``time_col``, ``target_col``,
            and any columns listed in ``covariates``.
            Must contain both pre- and post-intervention observations.
        intervention_date
            The first date/time of the post-intervention period. All
            observations with ``time_col >= intervention_date`` are
            treated as post-intervention.

        Returns
        -------
        CausalImpact
            Self, for chaining.

        """
        from scipy.stats import norm

        z = norm.ppf(1 - (1 - self.coverage) / 2)

        self._intervention_date = intervention_date
        self._resolved_roles = _validate_covariate_roles(self.covariates, self.covariate_role)
        sorted_df = df.sort(self.id_col, self.time_col)

        # Identify "always" covariates for post-period projection
        always_covs = [c for c, r in self._resolved_roles.items() if r == "always"]

        for group_id, group_df in sorted_df.group_by(self.id_col, maintain_order=True):
            gid = group_id[0]

            pre_df = group_df.filter(pl.col(self.time_col) < intervention_date)
            post_df = group_df.filter(pl.col(self.time_col) >= intervention_date)

            if len(pre_df) < 3:
                raise ValueError(
                    f"Series {gid!r}: pre-intervention period has {len(pre_df)} " f"observations, need at least 3."
                )
            if len(post_df) == 0:
                raise ValueError(
                    f"Series {gid!r}: no post-intervention observations found. "
                    f"Check that intervention_date={intervention_date} is within "
                    f"the data range."
                )

            pre_y = pre_df[self.target_col].to_numpy().astype(np.float64)
            post_y = post_df[self.target_col].to_numpy().astype(np.float64)

            # Covariate regression
            reg_post_contribution = np.zeros(len(post_y))
            bsts_pre_y = pre_y

            if self.covariates:
                # Validate pre-period covariates have no missing values
                for cov in self.covariates:
                    if cov not in group_df.columns:
                        raise ValueError(f"Series {gid!r}: covariate {cov!r} not found in DataFrame.")
                    pre_nulls = pre_df[cov].null_count()
                    if pre_nulls > 0:
                        raise ValueError(
                            f"Series {gid!r}: covariate {cov!r} " f"has {pre_nulls} missing values in pre-period."
                        )

                # Build covariate matrices
                all_covs = list(self.covariates)
                pre_X = pre_df.select(all_covs).to_numpy().astype(np.float64)
                beta, X_mean, _y_mean = _fit_regression(pre_y, pre_X)

                # Remove demeaned regression effect from pre-period target
                # (level/trend stay for BSTS to model)
                bsts_pre_y = pre_y - (pre_X - X_mean) @ beta

                # Post-period: only "always" covariates contribute
                if always_covs:
                    post_X_always = post_df.select(always_covs).to_numpy().astype(np.float64)
                    always_idx = [all_covs.index(c) for c in always_covs]
                    beta_always = beta[always_idx]
                    X_mean_always = X_mean[always_idx]
                    reg_post_contribution = (post_X_always - X_mean_always) @ beta_always

            model = BSTS(
                trend=self.trend,
                seasonal=self.seasonal,
                sigma_obs=self.sigma_obs,
                sigma_level=self.sigma_level,
                sigma_trend=self.sigma_trend,
                sigma_seasonal=self.sigma_seasonal,
            )

            bsts_result = model.forecast(bsts_pre_y, h=len(post_y))

            assert bsts_result.forecast is not None
            assert bsts_result.forecast_var is not None

            # Counterfactual = BSTS forecast + regression from "always" covariates
            counterfactual = bsts_result.forecast + reg_post_contribution
            cf_std = np.sqrt(np.maximum(bsts_result.forecast_var, 0.0))
            cf_lower = counterfactual - z * cf_std
            cf_upper = counterfactual + z * cf_std

            # Pointwise effect
            point_effect = post_y - counterfactual
            effect_lower = post_y - cf_upper  # lower effect when cf is high
            effect_upper = post_y - cf_lower  # upper effect when cf is low

            # Cumulative effect
            cum_effect = np.cumsum(point_effect)
            cum_lower = np.cumsum(effect_lower)
            cum_upper = np.cumsum(effect_upper)

            # Total effect
            total = float(np.sum(point_effect))
            total_lower = float(np.sum(effect_lower))
            total_upper = float(np.sum(effect_upper))

            # Relative effect
            cf_sum = float(np.sum(counterfactual))
            if abs(cf_sum) > 1e-10:
                rel = total / cf_sum
                rel_lo = total_lower / cf_sum
                rel_hi = total_upper / cf_sum
                # Division by negative cf_sum flips ordering
                rel_lower = min(rel_lo, rel_hi)
                rel_upper = max(rel_lo, rel_hi)
            else:
                rel = rel_lower = rel_upper = 0.0

            # Pre-period diagnostics (reuse smoothed states from forecast)
            kr = bsts_result.kalman_result
            assert kr.smoothed_states is not None
            assert kr.smoothed_covs is not None
            _, H_mat, _, R_mat = model._build_system()
            pre_fitted_bsts = np.array([float((H_mat @ kr.smoothed_states[t]).item()) for t in range(len(bsts_pre_y))])
            # Add back demeaned regression contribution for full pre-period fit
            if self.covariates:
                pre_X_all = pre_df.select(list(self.covariates)).to_numpy().astype(np.float64)
                pre_fitted = pre_fitted_bsts + (pre_X_all - X_mean) @ beta
            else:
                pre_fitted = pre_fitted_bsts
            pre_residuals = pre_y - pre_fitted
            pre_mape = float(np.mean(np.abs(pre_residuals / np.where(np.abs(pre_y) > 1e-10, pre_y, 1.0))))

            # Pre-period coverage: fraction of obs inside credible interval
            pre_fitted_var = np.array(
                [float((H_mat @ kr.smoothed_covs[t] @ H_mat.T + R_mat).item()) for t in range(len(bsts_pre_y))]
            )
            pre_std = np.sqrt(np.maximum(pre_fitted_var, 0.0))
            in_interval = np.abs(pre_residuals) <= z * pre_std
            pre_coverage = float(np.mean(in_interval))

            result = CausalImpactResult(
                point_effect=point_effect,
                point_effect_lower=effect_lower,
                point_effect_upper=effect_upper,
                cumulative_effect=cum_effect,
                cumulative_effect_lower=cum_lower,
                cumulative_effect_upper=cum_upper,
                total_effect=total,
                total_effect_lower=total_lower,
                total_effect_upper=total_upper,
                relative_effect=rel,
                relative_effect_lower=rel_lower,
                relative_effect_upper=rel_upper,
                counterfactual=counterfactual,
                counterfactual_lower=cf_lower,
                counterfactual_upper=cf_upper,
                observed_post=post_y,
                bsts_result=bsts_result,
                pre_mape=pre_mape,
                pre_coverage=pre_coverage,
            )

            state = _FitState(
                bsts_model=model,
                pre_y=pre_y,
                post_y=post_y,
                pre_len=len(pre_y),
                post_len=len(post_y),
                result=result,
            )
            self._states[gid] = state

        self.is_fitted_ = True
        return self

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


def causal_impact(
    df: pl.DataFrame,
    intervention_date: date | datetime,
    trend: str = "local_linear",
    seasonal: int | None = None,
    sigma_obs: float = 1.0,
    sigma_level: float = 0.1,
    sigma_trend: float = 0.01,
    sigma_seasonal: float = 0.01,
    coverage: float = 0.9,
    covariates: list[str] | None = None,
    covariate_role: dict[str, Literal["pre_only", "always"]] | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> dict[Any, CausalImpactResult]:
    """Estimate the causal effect of an intervention on time series.

    Convenience function wrapping :class:`CausalImpact`.

    Parameters
    ----------
    df
        Panel DataFrame.
    intervention_date
        First date of the post-intervention period.
    trend, seasonal, sigma_obs, sigma_level, sigma_trend, sigma_seasonal
        BSTS model configuration (see :class:`CausalImpact`).
    coverage
        Credible interval coverage.
    covariates
        Column names of exogenous covariates.
    covariate_role
        Mapping from covariate name to ``"pre_only"`` or ``"always"``.
    id_col, time_col, target_col
        Column names.

    Returns
    -------
    dict[Any, CausalImpactResult]
        Mapping from series ID to result.

    """
    ci = CausalImpact(
        trend=trend,
        seasonal=seasonal,
        sigma_obs=sigma_obs,
        sigma_level=sigma_level,
        sigma_trend=sigma_trend,
        sigma_seasonal=sigma_seasonal,
        coverage=coverage,
        covariates=covariates,
        covariate_role=covariate_role,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
    )
    ci.fit(df, intervention_date=intervention_date)
    return ci.results()
