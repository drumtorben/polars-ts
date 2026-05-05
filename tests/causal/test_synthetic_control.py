from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.causal.synthetic_control import (
    SyntheticControl,
    SyntheticControlResult,
    synthetic_control,
)


class TestSyntheticControlValidation:
    def test_result_before_fit(self):
        sc = SyntheticControl()
        with pytest.raises(RuntimeError, match="fit"):
            _ = sc.result

    def test_to_frame_before_fit(self):
        sc = SyntheticControl()
        with pytest.raises(RuntimeError, match="fit"):
            sc.to_frame()

    def test_missing_treated_id(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        with pytest.raises(ValueError, match="not found"):
            sc.fit(sc_panel_df, treated_id="missing", intervention_date=sc_intervention_date)

    def test_no_donors(self, sc_intervention_date):
        base = date(2024, 1, 1)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 60,
                "ds": [base + timedelta(days=i) for i in range(60)],
                "y": list(range(60)),
            }
        )
        sc = SyntheticControl()
        with pytest.raises(ValueError, match="No donor"):
            sc.fit(df, treated_id="A", intervention_date=sc_intervention_date)

    def test_short_pre_period(self, sc_panel_df):
        sc = SyntheticControl()
        # Intervention at second observation
        early_date = date(2024, 1, 2)
        with pytest.raises(ValueError, match="Pre-intervention"):
            sc.fit(sc_panel_df, treated_id="treated", intervention_date=early_date)

    def test_no_post_period(self, sc_panel_df):
        sc = SyntheticControl()
        late_date = date(2025, 12, 31)
        with pytest.raises(ValueError, match="post-intervention"):
            sc.fit(sc_panel_df, treated_id="treated", intervention_date=late_date)


class TestSyntheticControlFit:
    def test_basic_fit(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        result = sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        assert result is sc
        assert sc.is_fitted_

    def test_result_type(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        assert isinstance(sc.result, SyntheticControlResult)

    def test_weights_valid(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert np.all(r.weights >= 0)
        np.testing.assert_allclose(np.sum(r.weights), 1.0, atol=1e-6)

    def test_donor_ids(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert set(r.donor_ids) == {"D1", "D2", "D3"}

    def test_effect_direction(self, sc_panel_df, sc_intervention_date):
        """Treatment adds +8, so total_effect should be positive."""
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        assert sc.result.total_effect > 0

    def test_effect_arrays_shape(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        n_post = 15
        assert len(r.point_effect) == n_post
        assert len(r.point_effect_lower) == n_post
        assert len(r.point_effect_upper) == n_post

    def test_counterfactual_full_length(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        n_total = 50 + 15  # pre + post
        assert len(r.counterfactual) == n_total

    def test_interval_ordering(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert r.total_effect_lower <= r.total_effect
        assert r.total_effect <= r.total_effect_upper

    def test_pre_rmse(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        assert sc.result.pre_rmse >= 0.0

    def test_specific_donors(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(
            sc_panel_df,
            treated_id="treated",
            intervention_date=sc_intervention_date,
            donor_ids=["D1", "D2"],
        )
        r = sc.result
        assert set(r.donor_ids) == {"D1", "D2"}


class TestSyntheticControlToFrame:
    def test_to_frame_columns(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        frame = sc.to_frame()
        expected = {
            "step",
            "observed",
            "counterfactual",
            "counterfactual_lower",
            "counterfactual_upper",
            "point_effect",
            "point_effect_lower",
            "point_effect_upper",
        }
        assert set(frame.columns) == expected

    def test_to_frame_rows(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        frame = sc.to_frame()
        assert len(frame) == 15
        assert frame["step"].to_list() == list(range(1, 16))


class TestSyntheticControlPlacebo:
    def test_placebo_test(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        placebo = sc.placebo_test(sc_panel_df, intervention_date=sc_intervention_date)
        assert "unit_id" in placebo.columns
        assert "is_treated" in placebo.columns
        assert "total_effect" in placebo.columns
        assert "pre_rmse" in placebo.columns
        # Should have treated + donors
        assert len(placebo) >= 2
        # Treated unit should be marked
        treated_row = placebo.filter(pl.col("is_treated"))
        assert len(treated_row) == 1

    def test_placebo_treated_effect_largest(self, sc_panel_df, sc_intervention_date):
        """The actual treated unit should have a larger effect than placebos."""
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        placebo = sc.placebo_test(sc_panel_df, intervention_date=sc_intervention_date)
        treated_effect = placebo.filter(pl.col("is_treated"))["total_effect"][0]
        placebo_effects = placebo.filter(~pl.col("is_treated"))["total_effect"]
        # Treated effect should be larger than most placebos
        assert treated_effect > placebo_effects.mean()


class TestSyntheticControlConvenience:
    def test_convenience_function(self, sc_panel_df, sc_intervention_date):
        result = synthetic_control(
            sc_panel_df,
            treated_id="treated",
            intervention_date=sc_intervention_date,
        )
        assert isinstance(result, SyntheticControlResult)
        assert result.total_effect > 0


class TestSyntheticControlSingleDonor:
    def test_single_donor(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(
            sc_panel_df,
            treated_id="treated",
            intervention_date=sc_intervention_date,
            donor_ids=["D1"],
        )
        r = sc.result
        np.testing.assert_allclose(r.weights, [1.0])
        assert len(r.donor_ids) == 1

    def test_observed_post_on_result(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert len(r.observed_post) == 15
        np.testing.assert_allclose(r.point_effect, r.observed_post - r.counterfactual[-15:])


class TestSyntheticControlCoverage:
    def test_wider_coverage(self, sc_panel_df, sc_intervention_date):
        sc_90 = SyntheticControl(coverage=0.9)
        sc_90.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)

        sc_99 = SyntheticControl(coverage=0.99)
        sc_99.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)

        width_90 = sc_90.result.counterfactual_upper - sc_90.result.counterfactual_lower
        width_99 = sc_99.result.counterfactual_upper - sc_99.result.counterfactual_lower
        # Post-period intervals should be wider for 99%
        assert np.all(width_99[-15:] >= width_90[-15:] - 1e-10)


class TestSyntheticControlCovariates:
    def test_fit_with_covariates(self, sc_panel_cov_df, sc_intervention_date):
        sc = SyntheticControl(
            covariates=["temperature"],
            covariate_role={"temperature": "always"},
        )
        sc.fit(sc_panel_cov_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert r.total_effect > 0
        assert len(r.point_effect) == 15

    def test_covariate_balance_diagnostics(self, sc_panel_cov_df, sc_intervention_date):
        sc = SyntheticControl(
            covariates=["temperature"],
            covariate_role={"temperature": "always"},
        )
        sc.fit(sc_panel_cov_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert r.covariate_balance is not None
        assert "temperature" in r.covariate_balance
        balance = r.covariate_balance["temperature"]
        assert "treated_mean" in balance
        assert "synthetic_mean" in balance
        assert "abs_diff" in balance
        assert balance["abs_diff"] >= 0.0

    def test_no_covariates_no_balance(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl()
        sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)
        assert sc.result.covariate_balance is None

    def test_pre_only_covariates(self, sc_panel_cov_df, sc_intervention_date):
        sc = SyntheticControl(
            covariates=["temperature"],
            covariate_role={"temperature": "pre_only"},
        )
        sc.fit(sc_panel_cov_df, treated_id="treated", intervention_date=sc_intervention_date)
        r = sc.result
        assert r.total_effect > 0

    def test_missing_covariate_column_raises(self, sc_panel_df, sc_intervention_date):
        sc = SyntheticControl(covariates=["nonexistent"])
        with pytest.raises(ValueError, match="nonexistent"):
            sc.fit(sc_panel_df, treated_id="treated", intervention_date=sc_intervention_date)

    def test_warning_no_role(self, sc_panel_cov_df, sc_intervention_date):
        sc = SyntheticControl(covariates=["temperature"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc.fit(sc_panel_cov_df, treated_id="treated", intervention_date=sc_intervention_date)
            role_warnings = [x for x in w if "covariate_role" in str(x.message)]
            assert len(role_warnings) >= 1

    def test_convenience_with_covariates(self, sc_panel_cov_df, sc_intervention_date):
        result = synthetic_control(
            sc_panel_cov_df,
            treated_id="treated",
            intervention_date=sc_intervention_date,
            covariates=["temperature"],
            covariate_role={"temperature": "always"},
        )
        assert isinstance(result, SyntheticControlResult)
        assert result.total_effect > 0
        assert result.covariate_balance is not None

    def test_placebo_with_covariates(self, sc_panel_cov_df, sc_intervention_date):
        sc = SyntheticControl(
            covariates=["temperature"],
            covariate_role={"temperature": "always"},
        )
        sc.fit(sc_panel_cov_df, treated_id="treated", intervention_date=sc_intervention_date)
        placebo = sc.placebo_test(sc_panel_cov_df, intervention_date=sc_intervention_date)
        assert len(placebo) >= 2
