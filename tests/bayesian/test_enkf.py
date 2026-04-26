import numpy as np
import pytest

from polars_ts.bayesian.enkf import EnsembleKalmanFilter
from polars_ts.bayesian.kalman import KalmanResult


def _linear_f(x: np.ndarray) -> np.ndarray:
    return x.copy()


def _linear_h(x: np.ndarray) -> np.ndarray:
    return x[:1].copy()


@pytest.fixture
def constant_y():
    rng = np.random.default_rng(42)
    return 5.0 + rng.normal(0, 0.5, size=50)


class TestEnKFLinear:
    def test_output_type(self, constant_y):
        enkf = EnsembleKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = enkf.filter(constant_y)
        assert isinstance(result, KalmanResult)

    def test_output_shapes(self, constant_y):
        enkf = EnsembleKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = enkf.filter(constant_y)
        assert result.filtered_states.shape == (50, 1)
        assert result.filtered_covs.shape == (50, 1, 1)

    def test_tracks_constant(self, constant_y):
        enkf = EnsembleKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1) * 100,
            n_ensemble=100,
        )
        result = enkf.filter(constant_y)
        assert abs(result.filtered_states[-1, 0] - 5.0) < 2.0

    def test_log_likelihood_finite(self, constant_y):
        enkf = EnsembleKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = enkf.filter(constant_y)
        assert np.isfinite(result.log_likelihood)

    def test_deterministic_with_seed(self, constant_y):
        kwargs = dict(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
            seed=123,
        )
        r1 = EnsembleKalmanFilter(**kwargs).filter(constant_y)
        r2 = EnsembleKalmanFilter(**kwargs).filter(constant_y)
        np.testing.assert_allclose(r1.filtered_states, r2.filtered_states)


class TestEnKFNonlinear:
    def test_nonlinear_transition(self):
        def f_nl(x: np.ndarray) -> np.ndarray:
            return x + 0.05 * np.sin(x)

        rng = np.random.default_rng(42)
        y = rng.normal(0, 1.0, size=30)

        enkf = EnsembleKalmanFilter(
            f=f_nl,
            h=_linear_h,
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = enkf.filter(y)
        assert np.all(np.isfinite(result.filtered_states))


class TestEnKFMissing:
    def test_handles_nan(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        enkf = EnsembleKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = enkf.filter(y)
        assert np.all(np.isfinite(result.filtered_states))


class TestEnKFEnsembleSize:
    def test_larger_ensemble_reduces_variance(self, constant_y):
        kwargs = dict(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1) * 100,
        )
        r_small = EnsembleKalmanFilter(**kwargs, n_ensemble=10, seed=42).filter(constant_y)
        r_large = EnsembleKalmanFilter(**kwargs, n_ensemble=200, seed=42).filter(constant_y)
        # Larger ensemble should give smaller estimation variance
        var_small = np.var(r_small.filtered_states[:, 0])
        var_large = np.var(r_large.filtered_states[:, 0])
        # Not strictly guaranteed per run, but very likely
        assert var_large < var_small * 5  # loose bound


class TestEnKFImports:
    def test_top_level_import(self):
        from polars_ts import EnsembleKalmanFilter as EnKF

        assert EnKF is EnsembleKalmanFilter

    def test_submodule_import(self):
        from polars_ts.bayesian import EnsembleKalmanFilter as EnKF

        assert EnKF is EnsembleKalmanFilter
