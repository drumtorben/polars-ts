from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.causal.causal_impact import CausalImpact, CausalImpactResult, causal_impact


class TestCausalImpactValidation:
    def test_fit_before_results(self):
        ci = CausalImpact()
        with pytest.raises(RuntimeError, match="fit"):
            ci.results()

    def test_fit_before_summary(self):
        ci = CausalImpact()
        with pytest.raises(RuntimeError, match="fit"):
            ci.summary()

    def test_fit_before_to_frame(self):
        ci = CausalImpact()
        with pytest.raises(RuntimeError, match="fit"):
            ci.to_frame()

    def test_short_pre_period(self, intervention_date):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A", "A", "A"],
                "ds": [
                    intervention_date - timedelta(days=2),
                    intervention_date - timedelta(days=1),
                    intervention_date,
                    intervention_date + timedelta(days=1),
                ],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        ci = CausalImpact()
        with pytest.raises(ValueError, match="pre-intervention"):
            ci.fit(df, intervention_date=intervention_date)

    def test_no_post_period(self):
        base = date(2024, 1, 1)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 10,
                "ds": [base + timedelta(days=i) for i in range(10)],
                "y": list(range(10)),
            }
        )
        ci = CausalImpact()
        with pytest.raises(ValueError, match="post-intervention"):
            ci.fit(df, intervention_date=base + timedelta(days=100))


class TestCausalImpactFit:
    def test_basic_fit(self, causal_df, intervention_date):
        ci = CausalImpact()
        result = ci.fit(causal_df, intervention_date=intervention_date)
        assert result is ci
        assert ci.is_fitted_

    def test_results_dict(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        results = ci.results()
        assert "A" in results
        assert isinstance(results["A"], CausalImpactResult)

    def test_effect_direction(self, causal_df, intervention_date):
        """Treatment adds +10, so total_effect should be positive."""
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.total_effect > 0

    def test_credible_interval_ordering(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.total_effect_lower <= r.total_effect
        assert r.total_effect <= r.total_effect_upper
        assert np.all(r.counterfactual_lower <= r.counterfactual)
        assert np.all(r.counterfactual <= r.counterfactual_upper)

    def test_effect_arrays_shape(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        n_post = 20
        assert len(r.point_effect) == n_post
        assert len(r.point_effect_lower) == n_post
        assert len(r.point_effect_upper) == n_post
        assert len(r.cumulative_effect) == n_post
        assert len(r.observed_post) == n_post
        assert len(r.counterfactual) == n_post

    def test_pre_diagnostics(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.pre_mape >= 0.0
        assert 0.0 <= r.pre_coverage <= 1.0

    def test_cumulative_is_cumsum(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        np.testing.assert_allclose(r.cumulative_effect, np.cumsum(r.point_effect))


class TestCausalImpactSummary:
    def test_summary_columns(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        summary = ci.summary()
        expected = {
            "unique_id",
            "total_effect",
            "total_effect_lower",
            "total_effect_upper",
            "relative_effect",
            "relative_effect_lower",
            "relative_effect_upper",
            "pre_mape",
            "pre_coverage",
        }
        assert set(summary.columns) == expected

    def test_summary_one_row(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        assert len(ci.summary()) == 1


class TestCausalImpactToFrame:
    def test_to_frame_columns(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        frame = ci.to_frame()
        expected = {
            "unique_id",
            "step",
            "observed",
            "counterfactual",
            "counterfactual_lower",
            "counterfactual_upper",
            "point_effect",
            "point_effect_lower",
            "point_effect_upper",
            "cumulative_effect",
            "cumulative_effect_lower",
            "cumulative_effect_upper",
        }
        assert set(frame.columns) == expected

    def test_to_frame_rows(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        frame = ci.to_frame()
        assert len(frame) == 20
        assert frame["step"].to_list() == list(range(1, 21))


class TestCausalImpactPlacebo:
    def test_placebo_test(self, causal_df, intervention_date):
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        # Placebo at midpoint of pre-period
        placebo_date = date(2024, 1, 1) + timedelta(days=30)
        placebo_result = ci.placebo_test(causal_df, placebo_date=placebo_date)
        assert "total_effect" in placebo_result.columns
        assert len(placebo_result) == 1

    def test_placebo_effect_near_zero(self, causal_df, intervention_date):
        """Placebo effect should be small since there is no treatment before intervention."""
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        placebo_date = date(2024, 1, 1) + timedelta(days=30)
        placebo_result = ci.placebo_test(causal_df, placebo_date=placebo_date)
        # Placebo total effect should be much smaller than actual effect
        r = ci.results()["A"]
        placebo_effect = abs(placebo_result["total_effect"][0])
        assert placebo_effect < abs(r.total_effect)

    def test_placebo_before_fit_raises(self):
        ci = CausalImpact()
        with pytest.raises(RuntimeError, match="fit"):
            ci.placebo_test(pl.DataFrame(), placebo_date=date(2024, 1, 15))


class TestCausalImpactMultiGroup:
    def test_multi_group(self, intervention_date):
        rng = np.random.default_rng(42)
        n_pre, n_post = 60, 20
        n = n_pre + n_post
        base = date(2024, 1, 1)
        dates = [base + timedelta(days=i) for i in range(n)]
        rows = []
        for uid in ["A", "B"]:
            y = 100.0 + rng.normal(0, 1, n)
            y[n_pre:] += 5.0
            for i in range(n):
                rows.append({"unique_id": uid, "ds": dates[i], "y": float(y[i])})
        df = pl.DataFrame(rows)

        ci = CausalImpact()
        ci.fit(df, intervention_date=intervention_date)
        results = ci.results()
        assert "A" in results
        assert "B" in results
        assert len(ci.summary()) == 2


class TestCausalImpactConvenience:
    def test_convenience_function(self, causal_df, intervention_date):
        results = causal_impact(causal_df, intervention_date=intervention_date)
        assert "A" in results
        r = results["A"]
        assert isinstance(r, CausalImpactResult)
        assert r.total_effect > 0


class TestCausalImpactCoverage:
    def test_wider_coverage(self, causal_df, intervention_date):
        ci_90 = CausalImpact(coverage=0.9)
        ci_90.fit(causal_df, intervention_date=intervention_date)
        r90 = ci_90.results()["A"]

        ci_99 = CausalImpact(coverage=0.99)
        ci_99.fit(causal_df, intervention_date=intervention_date)
        r99 = ci_99.results()["A"]

        # 99% interval should be wider than 90%
        width_90 = r90.counterfactual_upper - r90.counterfactual_lower
        width_99 = r99.counterfactual_upper - r99.counterfactual_lower
        assert np.all(width_99 >= width_90 - 1e-10)


class TestCausalImpactLevelTrend:
    def test_level_trend(self, causal_df, intervention_date):
        ci = CausalImpact(trend="level")
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.total_effect > 0
        assert len(r.point_effect) == 20


class TestCausalImpactCovariates:
    def test_fit_with_covariates(self, causal_cov_df, intervention_date):
        ci = CausalImpact(
            covariates=["weather", "demand"],
            covariate_role={"weather": "always", "demand": "pre_only"},
        )
        ci.fit(causal_cov_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.total_effect > 0
        assert len(r.point_effect) == 20

    def test_pre_only_excluded_from_counterfactual(self, causal_cov_df, intervention_date):
        """With 'pre_only' covariates, post-treatment bias is avoided."""
        ci_with_role = CausalImpact(
            covariates=["weather", "demand"],
            covariate_role={"weather": "always", "demand": "pre_only"},
        )
        ci_with_role.fit(causal_cov_df, intervention_date=intervention_date)
        r_role = ci_with_role.results()["A"]

        # demand shifts +5 after treatment; if included as "always",
        # it would inflate the counterfactual and shrink the effect
        ci_all_always = CausalImpact(
            covariates=["weather", "demand"],
            covariate_role={"weather": "always", "demand": "always"},
        )
        ci_all_always.fit(causal_cov_df, intervention_date=intervention_date)
        r_always = ci_all_always.results()["A"]

        # pre_only correctly excludes demand from post counterfactual,
        # so it should detect a larger effect than when demand biases it
        assert r_role.total_effect > r_always.total_effect

    def test_no_covariates_backward_compat(self, causal_df, intervention_date):
        """Without covariates, behavior is unchanged."""
        ci = CausalImpact()
        ci.fit(causal_df, intervention_date=intervention_date)
        r = ci.results()["A"]
        assert r.total_effect > 0

    def test_missing_covariate_column_raises(self, causal_df, intervention_date):
        ci = CausalImpact(covariates=["nonexistent"])
        with pytest.raises(ValueError, match="nonexistent"):
            ci.fit(causal_df, intervention_date=intervention_date)

    def test_warning_no_role_specified(self, causal_cov_df, intervention_date):
        ci = CausalImpact(covariates=["weather", "demand"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ci.fit(causal_cov_df, intervention_date=intervention_date)
            role_warnings = [x for x in w if "covariate_role" in str(x.message)]
            assert len(role_warnings) >= 1

    def test_warning_all_always(self, causal_cov_df, intervention_date):
        ci = CausalImpact(
            covariates=["weather", "demand"],
            covariate_role={"weather": "always", "demand": "always"},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ci.fit(causal_cov_df, intervention_date=intervention_date)
            all_always_warnings = [x for x in w if "All covariates" in str(x.message)]
            assert len(all_always_warnings) == 1

    def test_convenience_with_covariates(self, causal_cov_df, intervention_date):
        results = causal_impact(
            causal_cov_df,
            intervention_date=intervention_date,
            covariates=["weather"],
            covariate_role={"weather": "always"},
        )
        assert "A" in results
        assert results["A"].total_effect > 0

    def test_placebo_with_covariates(self, causal_cov_df, intervention_date):
        ci = CausalImpact(
            covariates=["weather"],
            covariate_role={"weather": "always"},
        )
        ci.fit(causal_cov_df, intervention_date=intervention_date)
        placebo_date = date(2024, 1, 1) + timedelta(days=30)
        placebo_result = ci.placebo_test(causal_cov_df, placebo_date=placebo_date)
        assert "total_effect" in placebo_result.columns
