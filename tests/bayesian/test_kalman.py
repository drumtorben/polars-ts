import numpy as np
import polars as pl
import pytest

from polars_ts.bayesian.kalman import KalmanFilter, KalmanResult, kalman_filter

# --- Local-level model helpers ---
# State: x_t (scalar level)
# Observation: y_t = x_t + v_t
# Transition: x_t = x_{t-1} + w_t


def _local_level_matrices(q: float = 1.0, r: float = 1.0):
    """Return system matrices for a local-level (random walk + noise) model."""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[q]])
    R = np.array([[r]])
    return F, H, Q, R


@pytest.fixture
def constant_series():
    """Series with constant mean + noise."""
    rng = np.random.default_rng(42)
    y = 5.0 + rng.normal(0, 0.5, size=100)
    return y


@pytest.fixture
def trend_series():
    """Series with linear trend + noise."""
    rng = np.random.default_rng(42)
    y = np.linspace(0, 10, 100) + rng.normal(0, 0.5, size=100)
    return y


class TestKalmanFilter:
    def test_filter_output_type(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        assert isinstance(result, KalmanResult)

    def test_filter_shapes(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        T = len(constant_series)
        assert result.filtered_states.shape == (T, 1)
        assert result.filtered_covs.shape == (T, 1, 1)
        assert result.predicted_states.shape == (T, 1)
        assert result.predicted_covs.shape == (T, 1, 1)

    def test_filter_no_smoothed(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        assert result.smoothed_states is None
        assert result.smoothed_covs is None

    def test_log_likelihood_finite(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        assert np.isfinite(result.log_likelihood)

    def test_filtered_state_tracks_constant(self, constant_series):
        F, H, Q, R = _local_level_matrices(q=0.01, r=1.0)
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        # After convergence, filtered state should be near 5.0
        assert abs(result.filtered_states[-1, 0] - 5.0) < 1.0

    def test_covariance_positive_definite(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(constant_series)
        for t in range(len(constant_series)):
            eigvals = np.linalg.eigvalsh(result.filtered_covs[t])
            assert np.all(eigvals > 0)

    def test_custom_initial_state(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R, x0=np.array([5.0]), P0=np.array([[0.1]]))
        result = kf.filter(constant_series)
        # With good initial guess, first filtered state should be near 5.0
        assert abs(result.filtered_states[0, 0] - 5.0) < 1.0


class TestKalmanSmoother:
    def test_smooth_output_shapes(self, constant_series):
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.smooth(constant_series)
        T = len(constant_series)
        assert result.smoothed_states is not None
        assert result.smoothed_covs is not None
        assert result.smoothed_states.shape == (T, 1)
        assert result.smoothed_covs.shape == (T, 1, 1)

    def test_smoothed_covariance_smaller(self, constant_series):
        """Smoothed covariance should be <= filtered covariance."""
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.smooth(constant_series)
        assert result.smoothed_covs is not None
        for t in range(len(constant_series)):
            diff = result.filtered_covs[t] - result.smoothed_covs[t]
            eigvals = np.linalg.eigvalsh(diff)
            assert np.all(eigvals >= -1e-10)

    def test_last_smoothed_equals_filtered(self, constant_series):
        """At the last timestep, smoothed = filtered."""
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.smooth(constant_series)
        assert result.smoothed_states is not None
        np.testing.assert_allclose(
            result.smoothed_states[-1],
            result.filtered_states[-1],
            atol=1e-10,
        )

    def test_smoothed_tracks_constant(self, constant_series):
        F, H, Q, R = _local_level_matrices(q=0.01, r=1.0)
        kf = KalmanFilter(F, H, Q, R)
        result = kf.smooth(constant_series)
        assert result.smoothed_states is not None
        mean_state = result.smoothed_states[:, 0].mean()
        assert abs(mean_state - 5.0) < 0.5


class TestMissingObservations:
    def test_missing_values_handled(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        result = kf.filter(y)
        assert np.all(np.isfinite(result.filtered_states))

    def test_missing_increases_uncertainty(self):
        y_complete = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_missing = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        F, H, Q, R = _local_level_matrices()
        kf = KalmanFilter(F, H, Q, R)
        r_complete = kf.filter(y_complete)
        r_missing = kf.filter(y_missing)
        # At t=2 (missing), uncertainty should be higher
        assert r_missing.filtered_covs[2, 0, 0] > r_complete.filtered_covs[2, 0, 0]


class TestMultivariate:
    def test_2d_state(self):
        """Local linear trend: state = [level, trend]."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([0.1, 0.01])
        R = np.array([[1.0]])

        rng = np.random.default_rng(42)
        y = np.linspace(0, 10, 50) + rng.normal(0, 1.0, size=50)

        kf = KalmanFilter(F, H, Q, R)
        result = kf.smooth(y)
        assert result.filtered_states.shape == (50, 2)
        assert result.smoothed_states is not None
        assert result.smoothed_states.shape == (50, 2)
        # Trend component should be positive
        assert result.smoothed_states[-1, 1] > 0


class TestKalmanFilterFunction:
    def test_panel_data(self):
        rng = np.random.default_rng(42)
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 50 + ["B"] * 50,
                "y": (5.0 + rng.normal(0, 0.5, 50)).tolist() + (10.0 + rng.normal(0, 0.5, 50)).tolist(),
            }
        )
        F, H, Q, R = _local_level_matrices(q=0.01, r=1.0)
        results = kalman_filter(df, F, H, Q, R)
        assert set(results.keys()) == {"A", "B"}
        assert results["A"].smoothed_states is not None
        assert results["B"].smoothed_states is not None

    def test_filter_only(self):
        df = pl.DataFrame({"unique_id": ["A"] * 20, "y": list(range(20))})
        F, H, Q, R = _local_level_matrices()
        results = kalman_filter(df, F, H, Q, R, smooth=False)
        assert results["A"].smoothed_states is None

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 10, "val": [float(i) for i in range(10)]})
        F, H, Q, R = _local_level_matrices()
        results = kalman_filter(df, F, H, Q, R, id_col="sid", target_col="val")
        assert "X" in results

    def test_series_independence(self):
        """Each series should be filtered independently."""
        rng = np.random.default_rng(42)
        y_a = 5.0 + rng.normal(0, 0.5, 30)
        df_single = pl.DataFrame({"unique_id": ["A"] * 30, "y": y_a.tolist()})
        df_panel = pl.DataFrame(
            {
                "unique_id": ["A"] * 30 + ["B"] * 30,
                "y": y_a.tolist() + rng.normal(0, 1, 30).tolist(),
            }
        )
        F, H, Q, R = _local_level_matrices(q=0.01, r=1.0)
        r_single = kalman_filter(df_single, F, H, Q, R)
        r_panel = kalman_filter(df_panel, F, H, Q, R)
        assert r_single["A"].smoothed_states is not None
        assert r_panel["A"].smoothed_states is not None
        np.testing.assert_allclose(
            r_single["A"].filtered_states,
            r_panel["A"].filtered_states,
            atol=1e-10,
        )


class TestImports:
    def test_top_level_import(self):
        from polars_ts import KalmanFilter as KF

        assert KF is KalmanFilter

    def test_functional_import(self):
        from polars_ts import kalman_filter as kf

        assert callable(kf)

    def test_submodule_import(self):
        from polars_ts.bayesian import KalmanFilter as KF

        assert KF is KalmanFilter
