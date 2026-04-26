import numpy as np
import pytest

from polars_ts.bayesian.kalman import KalmanResult
from polars_ts.bayesian.ukf import UnscentedKalmanFilter


def _linear_f(x: np.ndarray) -> np.ndarray:
    """Linear random-walk transition."""
    return x.copy()


def _linear_h(x: np.ndarray) -> np.ndarray:
    """Linear identity observation."""
    return x[:1].copy()


def _nonlinear_f(x: np.ndarray) -> np.ndarray:
    """Nonlinear transition: x + 0.05 * sin(x)."""
    return x + 0.05 * np.sin(x)


@pytest.fixture
def constant_y():
    """Constant signal + noise."""
    rng = np.random.default_rng(42)
    return 5.0 + rng.normal(0, 0.5, size=50)


class TestUKFLinear:
    def test_output_type(self, constant_y):
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(constant_y)
        assert isinstance(result, KalmanResult)

    def test_output_shapes(self, constant_y):
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(constant_y)
        assert result.filtered_states.shape == (50, 1)
        assert result.filtered_covs.shape == (50, 1, 1)
        assert result.predicted_states.shape == (50, 1)

    def test_tracks_constant(self, constant_y):
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1) * 100,
        )
        result = ukf.filter(constant_y)
        assert abs(result.filtered_states[-1, 0] - 5.0) < 1.5

    def test_log_likelihood_finite(self, constant_y):
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(constant_y)
        assert np.isfinite(result.log_likelihood)

    def test_covariance_positive_definite(self, constant_y):
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[0.01]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(constant_y)
        for t in range(len(constant_y)):
            eigvals = np.linalg.eigvalsh(result.filtered_covs[t])
            assert np.all(eigvals > 0)


class TestUKFNonlinear:
    def test_nonlinear_transition(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1.0, size=30)

        ukf = UnscentedKalmanFilter(
            f=_nonlinear_f,
            h=_linear_h,
            Q=np.array([[0.1]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(y)
        assert result.filtered_states.shape == (30, 1)
        assert np.all(np.isfinite(result.filtered_states))


class TestUKFMissing:
    def test_handles_nan(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ukf = UnscentedKalmanFilter(
            f=_linear_f,
            h=_linear_h,
            Q=np.array([[1.0]]),
            R=np.array([[1.0]]),
            x0=np.array([0.0]),
            P0=np.eye(1),
        )
        result = ukf.filter(y)
        assert np.all(np.isfinite(result.filtered_states))


class TestUKFMultivariate:
    def test_2d_state(self):
        def f_2d(x: np.ndarray) -> np.ndarray:
            return np.array([x[0] + x[1], x[1]])

        def h_2d(x: np.ndarray) -> np.ndarray:
            return np.array([x[0]])

        rng = np.random.default_rng(42)
        y = np.linspace(0, 10, 30) + rng.normal(0, 0.5, 30)

        ukf = UnscentedKalmanFilter(
            f=f_2d,
            h=h_2d,
            Q=np.diag([0.1, 0.01]),
            R=np.array([[1.0]]),
            x0=np.array([0.0, 0.1]),
            P0=np.eye(2),
        )
        result = ukf.filter(y)
        assert result.filtered_states.shape == (30, 2)


class TestUKFImports:
    def test_top_level_import(self):
        from polars_ts import UnscentedKalmanFilter as UKF

        assert UKF is UnscentedKalmanFilter

    def test_submodule_import(self):
        from polars_ts.bayesian import UnscentedKalmanFilter as UKF

        assert UKF is UnscentedKalmanFilter
