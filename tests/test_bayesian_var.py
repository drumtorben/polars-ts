"""Tests for Bayesian VAR model (#118)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from polars_ts.bayesian_var import (
    BayesianVAR,
    BayesianVARResult,
    MinnesotaPrior,
    NormalWishartPrior,
    _build_var_matrices,
    _estimate_sigma_from_ar,
    _minnesota_prior_precision,
    bayesian_var,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_var_data(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Generate a bivariate VAR(1) process with known dynamics."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = rng.normal()
    y[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + rng.normal(0, 0.5)
        y[t] = 0.2 * x[t - 1] + 0.6 * y[t - 1] + rng.normal(0, 0.5)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "ds": [base + timedelta(days=i) for i in range(n)],
            "x": x.tolist(),
            "y": y.tolist(),
        }
    )


def _make_grouped_var_data(n: int = 80, seed: int = 42) -> pl.DataFrame:
    """Generate grouped bivariate VAR data."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    base = date(2024, 1, 1)
    for gid in ["A", "B"]:
        x = np.zeros(n)
        y = np.zeros(n)
        x[0] = rng.normal()
        y[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.4 * x[t - 1] + 0.2 * y[t - 1] + rng.normal(0, 0.5)
            y[t] = 0.1 * x[t - 1] + 0.5 * y[t - 1] + rng.normal(0, 0.5)
        for t in range(n):
            rows.append(
                {
                    "unique_id": gid,
                    "ds": base + timedelta(days=t),
                    "x": float(x[t]),
                    "y": float(y[t]),
                }
            )
    return pl.DataFrame(rows)


def _make_three_var_data(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Generate a trivariate VAR process."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 3))
    data[0] = rng.normal(size=3)
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.1, 0.6]])
    for t in range(1, n):
        data[t] = A @ data[t - 1] + rng.normal(0, 0.3, 3)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "ds": [base + timedelta(days=i) for i in range(n)],
            "x": data[:, 0].tolist(),
            "y": data[:, 1].tolist(),
            "z": data[:, 2].tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Prior dataclasses
# ---------------------------------------------------------------------------


class TestMinnesotaPrior:
    def test_defaults(self):
        p = MinnesotaPrior()
        assert p.lambda1 == 0.2
        assert p.lambda2 == 0.5
        assert p.lambda3 == 1.0
        assert p.sigma_scale is None

    def test_custom(self):
        p = MinnesotaPrior(lambda1=0.5, lambda2=0.8)
        assert p.lambda1 == 0.5
        assert p.lambda2 == 0.8


class TestNormalWishartPrior:
    def test_defaults(self):
        p = NormalWishartPrior()
        assert p.B0 is None
        assert p.V0 is None
        assert p.tightness == 0.1

    def test_custom(self):
        B0 = np.eye(2, 3)
        p = NormalWishartPrior(B0=B0, nu0=10.0)
        assert p.nu0 == 10.0
        assert p.B0 is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestBuildVarMatrices:
    def test_shapes(self):
        data = np.random.randn(20, 2)
        X, Y = _build_var_matrices(data, p=1)
        assert X.shape == (19, 3)  # 19 rows, 2*1 + 1 intercept
        assert Y.shape == (19, 2)

    def test_shapes_p2(self):
        data = np.random.randn(20, 3)
        X, Y = _build_var_matrices(data, p=2)
        assert X.shape == (18, 7)  # 18 rows, 3*2 + 1 intercept
        assert Y.shape == (18, 3)

    def test_intercept_column(self):
        data = np.random.randn(10, 2)
        X, _ = _build_var_matrices(data, p=1)
        assert np.all(X[:, -1] == 1.0)


class TestEstimateSigma:
    def test_positive(self):
        data = np.random.randn(50, 3)
        sigmas = _estimate_sigma_from_ar(data, p=1)
        assert len(sigmas) == 3
        assert all(s > 0 for s in sigmas)

    def test_constant_series(self):
        data = np.ones((50, 2))
        sigmas = _estimate_sigma_from_ar(data, p=1)
        # Near-zero variance for constant series
        assert all(s < 0.01 for s in sigmas)


class TestMinnesotaPriorPrecision:
    def test_b0_random_walk(self):
        sigma_scale = np.array([1.0, 1.0])
        B0, V0_inv = _minnesota_prior_precision(2, 1, MinnesotaPrior(), sigma_scale)
        # Own first lag should be 1 on diagonal
        assert B0[0, 0] == 1.0
        assert B0[1, 1] == 1.0
        assert B0[0, 1] == 0.0

    def test_diffuse_intercept(self):
        sigma_scale = np.array([1.0, 1.0])
        _, V0_inv = _minnesota_prior_precision(2, 1, MinnesotaPrior(), sigma_scale)
        # Last element (intercept) should be very small (diffuse)
        assert V0_inv[-1] < 1e-4


# ---------------------------------------------------------------------------
# BayesianVAR — validation
# ---------------------------------------------------------------------------


class TestBayesianVARValidation:
    def test_too_few_columns(self):
        with pytest.raises(ValueError, match="at least 2"):
            BayesianVAR(target_cols=["x"])

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p must"):
            BayesianVAR(target_cols=["x", "y"], p=0)

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="prior"):
            BayesianVAR(target_cols=["x", "y"], prior="laplace")

    def test_invalid_inference(self):
        with pytest.raises(ValueError, match="inference"):
            BayesianVAR(target_cols=["x", "y"], inference="hmc")

    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            BayesianVAR(target_cols=["x", "y"], coverage=0.0)

    def test_predict_before_fit(self):
        model = BayesianVAR(target_cols=["x", "y"])
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(horizon=5)

    def test_irf_before_fit(self):
        model = BayesianVAR(target_cols=["x", "y"])
        with pytest.raises(RuntimeError, match="fit"):
            model.irf(steps=10)

    def test_invalid_horizon(self):
        model = BayesianVAR(target_cols=["x", "y"])
        model.fit(_make_var_data())
        with pytest.raises(ValueError, match="positive"):
            model.predict(horizon=0)

    def test_invalid_irf_steps(self):
        model = BayesianVAR(target_cols=["x", "y"])
        model.fit(_make_var_data())
        with pytest.raises(ValueError, match="positive"):
            model.irf(steps=0)

    def test_short_series(self):
        df = pl.DataFrame(
            {
                "ds": [date(2024, 1, 1)],
                "x": [1.0],
                "y": [2.0],
            }
        )
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        with pytest.raises(ValueError, match="more than"):
            model.fit(df)

    def test_irf_unknown_group(self):
        model = BayesianVAR(target_cols=["x", "y"])
        model.fit(_make_var_data())
        with pytest.raises(ValueError, match="not found"):
            model.irf(steps=10, gid="nonexistent")


# ---------------------------------------------------------------------------
# BayesianVAR — analytical inference, Minnesota prior
# ---------------------------------------------------------------------------


class TestBayesianVARAnalyticalMinnesota:
    def test_basic_fit_predict(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1, prior="minnesota")
        model.fit(df)
        result = model.predict(horizon=5)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "x_lower" in result.columns
        assert "x_upper" in result.columns
        assert "y_lower" in result.columns
        assert "y_upper" in result.columns
        assert len(result) == 5

    def test_credible_intervals_contain_mean(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        result = model.predict(horizon=5)
        for col in ["x", "y"]:
            assert (result[f"{col}_lower"] <= result[col]).all()
            assert (result[col] <= result[f"{col}_upper"]).all()

    def test_intervals_widen_with_horizon(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        result = model.predict(horizon=10)
        widths = (result["x_upper"] - result["x_lower"]).to_list()
        for i in range(len(widths) - 1):
            assert widths[i + 1] > widths[i]

    def test_p2(self):
        df = _make_var_data(n=150)
        model = BayesianVAR(target_cols=["x", "y"], p=2, prior="minnesota")
        model.fit(df)
        result = model.predict(horizon=5)
        assert len(result) == 5

    def test_three_variables(self):
        df = _make_three_var_data()
        model = BayesianVAR(target_cols=["x", "y", "z"], p=1)
        model.fit(df)
        result = model.predict(horizon=3)
        assert len(result) == 3
        for col in ["x", "y", "z"]:
            assert col in result.columns
            assert f"{col}_lower" in result.columns
            assert f"{col}_upper" in result.columns

    def test_custom_minnesota_prior(self):
        df = _make_var_data()
        mp = MinnesotaPrior(lambda1=0.5, lambda2=0.8, lambda3=2.0)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            minnesota_prior=mp,
        )
        model.fit(df)
        result = model.predict(horizon=3)
        assert len(result) == 3

    def test_higher_coverage_wider(self):
        df = _make_var_data()
        m90 = BayesianVAR(target_cols=["x", "y"], p=1, coverage=0.9)
        m90.fit(df)
        r90 = m90.predict(horizon=5)

        m50 = BayesianVAR(target_cols=["x", "y"], p=1, coverage=0.5)
        m50.fit(df)
        r50 = m50.predict(horizon=5)

        w90 = (r90["x_upper"] - r90["x_lower"]).mean()
        w50 = (r50["x_upper"] - r50["x_lower"]).mean()
        assert w90 > w50

    def test_forecasts_reasonable(self):
        df = _make_var_data(n=200)
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        result = model.predict(horizon=5)
        # Forecasts should be in a reasonable range
        for col in ["x", "y"]:
            vals = result[col].to_list()
            assert all(abs(v) < 50 for v in vals)


# ---------------------------------------------------------------------------
# BayesianVAR — analytical inference, Normal-Wishart prior
# ---------------------------------------------------------------------------


class TestBayesianVARAnalyticalNW:
    def test_basic(self):
        df = _make_var_data()
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            prior="normal_wishart",
        )
        model.fit(df)
        result = model.predict(horizon=5)
        assert len(result) == 5

    def test_custom_nw_prior(self):
        df = _make_var_data()
        nw = NormalWishartPrior(nu0=10.0, tightness=0.5)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            prior="normal_wishart",
            nw_prior=nw,
        )
        model.fit(df)
        result = model.predict(horizon=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# BayesianVAR — Gibbs sampling
# ---------------------------------------------------------------------------


class TestBayesianVARGibbs:
    def test_basic_fit_predict(self):
        df = _make_var_data(n=80)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=100,
            burn_in=50,
        )
        model.fit(df)
        result = model.predict(horizon=5)
        assert len(result) == 5
        assert "x_lower" in result.columns
        assert "x_upper" in result.columns

    def test_credible_intervals(self):
        df = _make_var_data(n=80)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=100,
            burn_in=50,
        )
        model.fit(df)
        result = model.predict(horizon=5)
        for col in ["x", "y"]:
            assert (result[f"{col}_lower"] <= result[f"{col}_upper"]).all()

    def test_posterior_samples_stored(self):
        df = _make_var_data(n=80)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=50,
            burn_in=25,
        )
        model.fit(df)
        result = model._results["__global__"]
        assert result.B_samples is not None
        assert result.B_samples.shape == (50, 2, 3)  # 50 samples, 2 vars, 2*1+1
        assert result.Sigma_samples is not None
        assert result.Sigma_samples.shape == (50, 2, 2)

    def test_gibbs_reproducible(self):
        df = _make_var_data(n=80)
        m1 = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=50,
            burn_in=25,
            seed=42,
        )
        m1.fit(df)
        r1 = m1.predict(horizon=3)

        m2 = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=50,
            burn_in=25,
            seed=42,
        )
        m2.fit(df)
        r2 = m2.predict(horizon=3)

        assert r1["x"].to_list() == pytest.approx(r2["x"].to_list())

    def test_gibbs_nw_prior(self):
        df = _make_var_data(n=80)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            prior="normal_wishart",
            inference="gibbs",
            n_samples=50,
            burn_in=25,
        )
        model.fit(df)
        result = model.predict(horizon=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Impulse Response Functions
# ---------------------------------------------------------------------------


class TestIRF:
    def test_analytical_irf(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        irf_df = model.irf(steps=10)
        assert "step" in irf_df.columns
        assert "impulse" in irf_df.columns
        assert "response" in irf_df.columns
        assert "irf" in irf_df.columns
        assert "irf_lower" in irf_df.columns
        assert "irf_upper" in irf_df.columns
        # 10 steps × 2 impulse × 2 response = 40 rows
        assert len(irf_df) == 40

    def test_irf_initial_shock(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        irf_df = model.irf(steps=10, shock_size=1.0)
        # At step 1, own impulse should be 1.0 (identity shock)
        own_shock = irf_df.filter((pl.col("step") == 1) & (pl.col("impulse") == "x") & (pl.col("response") == "x"))[
            "irf"
        ].to_list()
        assert own_shock[0] == pytest.approx(1.0)

    def test_irf_decays(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        irf_df = model.irf(steps=20)
        # For a stable VAR, IRF should decay
        x_to_x = irf_df.filter((pl.col("impulse") == "x") & (pl.col("response") == "x")).sort("step")["irf"].to_list()
        # Should decay toward 0 (absolute value decreases over time)
        assert abs(x_to_x[-1]) < abs(x_to_x[0])

    def test_irf_credible_bands(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        irf_df = model.irf(steps=10)
        assert (irf_df["irf_lower"] <= irf_df["irf_upper"]).all()

    def test_gibbs_irf(self):
        df = _make_var_data(n=80)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=50,
            burn_in=25,
        )
        model.fit(df)
        irf_df = model.irf(steps=10)
        assert len(irf_df) == 40
        assert (irf_df["irf_lower"] <= irf_df["irf_upper"]).all()

    def test_three_var_irf(self):
        df = _make_three_var_data()
        model = BayesianVAR(target_cols=["x", "y", "z"], p=1)
        model.fit(df)
        irf_df = model.irf(steps=5)
        # 5 steps × 3 impulse × 3 response = 45 rows
        assert len(irf_df) == 45

    def test_custom_shock_size(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        irf1 = model.irf(steps=5, shock_size=1.0)
        irf2 = model.irf(steps=5, shock_size=2.0)
        # Doubling shock size should double IRF
        vals1 = irf1["irf"].to_list()
        vals2 = irf2["irf"].to_list()
        for v1, v2 in zip(vals1, vals2, strict=False):
            assert v2 == pytest.approx(2 * v1, abs=1e-10)


# ---------------------------------------------------------------------------
# Multi-group support
# ---------------------------------------------------------------------------


class TestMultiGroup:
    def test_grouped_fit_predict(self):
        df = _make_grouped_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df, id_col="unique_id")
        result = model.predict(horizon=5, id_col="unique_id")
        assert "unique_id" in result.columns
        assert len(result) == 10  # 2 groups × 5 steps
        groups = result["unique_id"].unique().to_list()
        assert sorted(groups) == ["A", "B"]

    def test_grouped_irf(self):
        df = _make_grouped_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df, id_col="unique_id")
        irf_a = model.irf(steps=5, gid="A")
        irf_b = model.irf(steps=5, gid="B")
        assert len(irf_a) == 20  # 5 × 2 × 2
        assert len(irf_b) == 20
        # Different groups should produce different IRFs
        assert irf_a["irf"].to_list() != irf_b["irf"].to_list()

    def test_grouped_gibbs(self):
        df = _make_grouped_var_data(n=60)
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            inference="gibbs",
            n_samples=30,
            burn_in=15,
        )
        model.fit(df, id_col="unique_id")
        result = model.predict(horizon=3, id_col="unique_id")
        assert len(result) == 6  # 2 groups × 3 steps


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestBayesianVARFunction:
    def test_basic(self):
        df = _make_var_data()
        result = bayesian_var(df, target_cols=["x", "y"], horizon=5)
        assert len(result) == 5
        assert "x" in result.columns
        assert "x_lower" in result.columns

    def test_minnesota(self):
        df = _make_var_data()
        result = bayesian_var(
            df,
            target_cols=["x", "y"],
            horizon=3,
            prior="minnesota",
        )
        assert len(result) == 3

    def test_nw(self):
        df = _make_var_data()
        result = bayesian_var(
            df,
            target_cols=["x", "y"],
            horizon=3,
            prior="normal_wishart",
        )
        assert len(result) == 3

    def test_gibbs(self):
        df = _make_var_data(n=80)
        result = bayesian_var(
            df,
            target_cols=["x", "y"],
            horizon=3,
            inference="gibbs",
            n_samples=50,
            burn_in=25,
        )
        assert len(result) == 3

    def test_grouped(self):
        df = _make_grouped_var_data()
        result = bayesian_var(
            df,
            target_cols=["x", "y"],
            horizon=3,
            id_col="unique_id",
        )
        assert len(result) == 6
        assert "unique_id" in result.columns

    def test_custom_time_col(self):
        rng = np.random.default_rng(42)
        n = 50
        base = date(2024, 1, 1)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(days=i) for i in range(n)],
                "x": rng.normal(0, 1, n).tolist(),
                "y": rng.normal(0, 1, n).tolist(),
            }
        )
        result = bayesian_var(
            df,
            target_cols=["x", "y"],
            horizon=3,
            time_col="timestamp",
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


def test_top_level_imports():
    import polars_ts

    # bayesian_var function shares name with the submodule;
    # direct attribute access returns the module, so test callable access
    assert callable(polars_ts.bayesian_var.bayesian_var)
    assert polars_ts.BayesianVAR is BayesianVAR
    assert polars_ts.MinnesotaPrior is MinnesotaPrior
    assert polars_ts.NormalWishartPrior is NormalWishartPrior
    assert polars_ts.BayesianVARResult is BayesianVARResult


# ---------------------------------------------------------------------------
# BayesianVARResult dataclass
# ---------------------------------------------------------------------------


class TestBayesianVARResult:
    def test_fields(self):
        r = BayesianVARResult(
            B_post=np.eye(2, 3),
            Sigma_post=np.eye(2),
            target_cols=["x", "y"],
            p=1,
        )
        assert r.B_post.shape == (2, 3)
        assert r.B_samples is None
        assert r.Sigma_samples is None

    def test_with_samples(self):
        r = BayesianVARResult(
            B_post=np.eye(2, 3),
            Sigma_post=np.eye(2),
            B_samples=np.random.randn(10, 2, 3),
            Sigma_samples=np.random.randn(10, 2, 2),
            target_cols=["x", "y"],
            p=1,
        )
        assert r.B_samples.shape == (10, 2, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_fit_returns_self(self):
        model = BayesianVAR(target_cols=["x", "y"])
        returned = model.fit(_make_var_data())
        assert returned is model
        assert model.is_fitted_

    def test_large_horizon(self):
        df = _make_var_data()
        model = BayesianVAR(target_cols=["x", "y"], p=1)
        model.fit(df)
        result = model.predict(horizon=50)
        assert len(result) == 50

    def test_p3(self):
        df = _make_var_data(n=200)
        model = BayesianVAR(target_cols=["x", "y"], p=3)
        model.fit(df)
        result = model.predict(horizon=5)
        assert len(result) == 5

    def test_sigma_scale_provided(self):
        df = _make_var_data()
        mp = MinnesotaPrior(sigma_scale=np.array([1.0, 2.0]))
        model = BayesianVAR(
            target_cols=["x", "y"],
            p=1,
            minnesota_prior=mp,
        )
        model.fit(df)
        result = model.predict(horizon=3)
        assert len(result) == 3
