"""Tests for Gaussian Process regression (#120)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")

from polars_ts.bayesian.gp import (  # noqa: E402
    GaussianProcessTS,
    GPResult,
    Matern32Kernel,
    Matern52Kernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    SpectralMixtureKernel,
    SumKernel,
    gp_forecast,
    make_kernel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 40, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    values = 10.0 + 0.3 * t + rng.normal(0, 0.5, n)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


def _make_multi_df(n: int = 30, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = date(2024, 1, 1)
    vals_a = 10.0 + 0.2 * np.arange(n) + rng.normal(0, 0.3, n)
    vals_b = 20.0 - 0.1 * np.arange(n) + rng.normal(0, 0.3, n)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)] * 2,
            "y": vals_a.tolist() + vals_b.tolist(),
        }
    )


def _make_periodic_df(n: int = 60, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    values = 5.0 + 3.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.3, n)
    base = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


class TestKernels:
    def test_rbf_shape(self):
        k = RBFKernel()
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([1.5, 2.5])
        K = k(X1, X2)
        assert K.shape == (3, 2)

    def test_rbf_diagonal_is_variance(self):
        k = RBFKernel(variance=2.0)
        X = np.array([1.0, 2.0, 3.0])
        K = k(X, X)
        np.testing.assert_allclose(np.diag(K), 2.0)

    def test_rbf_symmetric(self):
        k = RBFKernel()
        X = np.array([1.0, 2.0, 3.0])
        K = k(X, X)
        np.testing.assert_allclose(K, K.T)

    def test_matern32_shape(self):
        k = Matern32Kernel()
        K = k(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert K.shape == (2, 2)

    def test_matern52_shape(self):
        k = Matern52Kernel()
        K = k(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert K.shape == (2, 2)

    def test_periodic_shape(self):
        k = PeriodicKernel(period=12.0)
        K = k(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        assert K.shape == (2, 3)

    def test_spectral_mixture_shape(self):
        k = SpectralMixtureKernel(n_mixtures=2)
        K = k(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert K.shape == (2, 2)

    def test_sum_kernel(self):
        k = RBFKernel() + PeriodicKernel()
        assert isinstance(k, SumKernel)
        X = np.array([1.0, 2.0])
        K = k(X, X)
        assert K.shape == (2, 2)

    def test_product_kernel(self):
        k = RBFKernel() * PeriodicKernel()
        assert isinstance(k, ProductKernel)
        X = np.array([1.0, 2.0])
        K = k(X, X)
        assert K.shape == (2, 2)

    def test_make_kernel_rbf(self):
        k = make_kernel("rbf")
        assert isinstance(k, RBFKernel)

    def test_make_kernel_unknown(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            make_kernel("invalid")

    def test_param_roundtrip(self):
        k = RBFKernel(variance=2.0, lengthscale=3.0)
        params = k.get_params()
        k2 = RBFKernel()
        k2.set_params(params)
        assert k2.variance == pytest.approx(2.0)
        assert k2.lengthscale == pytest.approx(3.0)

    def test_sum_kernel_params(self):
        k = RBFKernel() + Matern32Kernel()
        assert k.n_params() == 4
        params = k.get_params()
        assert len(params) == 4


# ---------------------------------------------------------------------------
# GaussianProcessTS validation
# ---------------------------------------------------------------------------


class TestGPValidation:
    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            GaussianProcessTS(coverage=0.0)

    def test_predict_before_fit(self):
        gp = GaussianProcessTS()
        with pytest.raises(RuntimeError, match="fit"):
            gp.predict(_make_df(), h=3)

    def test_predict_invalid_horizon(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_df())
        with pytest.raises(ValueError, match="positive"):
            gp.predict(_make_df(), h=0)

    def test_predict_unseen_group(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_df())
        df_pred = pl.DataFrame(
            {
                "unique_id": ["Z"] * 10,
                "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        with pytest.raises(ValueError, match="not seen"):
            gp.predict(df_pred, h=3)


# ---------------------------------------------------------------------------
# GP fit/predict
# ---------------------------------------------------------------------------


class TestGPFitPredict:
    def test_output_shape(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_df())
        result = gp.predict(_make_df(), h=5)
        assert result.columns == ["unique_id", "step", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 5

    def test_credible_intervals(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_df())
        result = gp.predict(_make_df(), h=5)
        assert (result["y_hat_lower"] < result["y_hat_upper"]).all()

    def test_intervals_widen(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_df())
        result = gp.predict(_make_df(), h=10)
        widths = (result["y_hat_upper"] - result["y_hat_lower"]).to_list()
        # GP uncertainty should generally increase with horizon
        assert widths[-1] >= widths[0]

    def test_multi_group(self):
        gp = GaussianProcessTS(optimize=False)
        gp.fit(_make_multi_df())
        result = gp.predict(_make_multi_df(), h=3)
        assert len(result) == 6
        assert sorted(result["unique_id"].unique().to_list()) == ["A", "B"]

    def test_fit_returns_self(self):
        gp = GaussianProcessTS(optimize=False)
        returned = gp.fit(_make_df())
        assert returned is gp
        assert gp.is_fitted_

    def test_optimized(self):
        gp = GaussianProcessTS(kernel="rbf", optimize=True)
        gp.fit(_make_df())
        result = gp.predict(_make_df(), h=3)
        assert len(result) == 3

    def test_constant_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 20,
                "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(20)],
                "y": [42.0] * 20,
            }
        )
        gp = GaussianProcessTS(optimize=False)
        gp.fit(df)
        result = gp.predict(df, h=3)
        assert all(abs(v - 42.0) < 5.0 for v in result["y_hat"].to_list())


# ---------------------------------------------------------------------------
# Different kernels
# ---------------------------------------------------------------------------


class TestDifferentKernels:
    def test_matern32(self):
        result = gp_forecast(_make_df(), h=3, kernel="matern32", optimize=False)
        assert len(result) == 3

    def test_matern52(self):
        result = gp_forecast(_make_df(), h=3, kernel="matern52", optimize=False)
        assert len(result) == 3

    def test_periodic(self):
        result = gp_forecast(_make_periodic_df(), h=6, kernel="periodic", optimize=False)
        assert len(result) == 6

    def test_spectral_mixture(self):
        result = gp_forecast(_make_df(), h=3, kernel="spectral_mixture", optimize=False)
        assert len(result) == 3

    def test_composite_kernel(self):
        kernel = RBFKernel() + PeriodicKernel(period=12.0)
        result = gp_forecast(_make_periodic_df(), h=6, kernel=kernel, optimize=False)
        assert len(result) == 6

    def test_product_kernel_forecast(self):
        kernel = RBFKernel() * PeriodicKernel(period=12.0)
        result = gp_forecast(_make_periodic_df(), h=6, kernel=kernel, optimize=False)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestGPForecastFunction:
    def test_basic(self):
        result = gp_forecast(_make_df(), h=3, optimize=False)
        assert result.columns == ["unique_id", "step", "y_hat", "y_hat_lower", "y_hat_upper"]
        assert len(result) == 3

    def test_multi_group(self):
        result = gp_forecast(_make_multi_df(), h=3, optimize=False)
        assert len(result) == 6

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "sid": ["X"] * 20,
                "ts": [date(2024, 1, 1) + timedelta(days=i) for i in range(20)],
                "val": [float(i) for i in range(20)],
            }
        )
        result = gp_forecast(df, h=3, optimize=False, id_col="sid", target_col="val", time_col="ts")
        assert "sid" in result.columns
        assert len(result) == 3

    def test_higher_coverage_wider(self):
        df = _make_df()
        r90 = gp_forecast(df, h=3, coverage=0.9, optimize=False)
        r50 = gp_forecast(df, h=3, coverage=0.5, optimize=False)
        w90 = (r90["y_hat_upper"] - r90["y_hat_lower"]).mean()
        w50 = (r50["y_hat_upper"] - r50["y_hat_lower"]).mean()
        assert w90 > w50


# ---------------------------------------------------------------------------
# GPResult dataclass
# ---------------------------------------------------------------------------


class TestGPResult:
    def test_fields(self):
        r = GPResult(
            kernel=RBFKernel(),
            noise_var=0.1,
            X_train=np.array([1.0, 2.0]),
            y_train=np.array([1.0, 2.0]),
        )
        assert r.noise_var == 0.1


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


def test_submodule_imports():
    from polars_ts.bayesian.gp import GaussianProcessTS as GP
    from polars_ts.bayesian.gp import gp_forecast as gpf

    assert GP is GaussianProcessTS
    assert gpf is gp_forecast
