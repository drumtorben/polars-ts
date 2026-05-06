"""Tests for KASBA PyO3 bindings (issue #192)."""

from __future__ import annotations

import numpy as np
import pytest
from polars_ts_rs.polars_ts_rs import kasba_fit, kasba_predict


@pytest.fixture
def sample_3d_array():
    """Create sample univariate data: 10 series, 1 channel, 20 timepoints."""
    rng = np.random.default_rng(42)
    # Two distinct clusters: linear up vs linear down
    cluster_a = np.array([np.arange(20, dtype=np.float64) + rng.normal(0, 0.1, 20) for _ in range(5)])
    cluster_b = np.array([np.arange(19, -1, -1, dtype=np.float64) + rng.normal(0, 0.1, 20) for _ in range(5)])
    data = np.concatenate([cluster_a, cluster_b], axis=0)
    # Reshape to (n_cases, n_channels, n_timepoints)
    return data.reshape(10, 1, 20)


@pytest.fixture
def multivariate_3d_array():
    """Create multivariate data: 12 series, 3 channels, 15 timepoints."""
    rng = np.random.default_rng(123)
    data = np.zeros((12, 3, 15), dtype=np.float64)
    for i in range(4):
        data[i] = rng.normal(0, 1, (3, 15))
    for i in range(4, 8):
        data[i] = rng.normal(5, 1, (3, 15))
    for i in range(8, 12):
        data[i] = rng.normal(-5, 1, (3, 15))
    return data


class TestKASBAFit:
    """Test kasba_fit binding."""

    def test_returns_correct_types(self, sample_3d_array):
        labels, centers, inertia, n_iter = kasba_fit(sample_3d_array, n_clusters=2)
        assert isinstance(labels, np.ndarray)
        assert isinstance(centers, np.ndarray)
        assert isinstance(inertia, float)
        assert isinstance(n_iter, int)

    def test_labels_shape_and_dtype(self, sample_3d_array):
        labels, _, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        assert labels.shape == (10,)
        assert labels.dtype == np.int64

    def test_centers_shape(self, sample_3d_array):
        _, centers, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        # Centers: (n_clusters, n_channels * n_timepoints)
        assert centers.shape == (2, 1 * 20)

    def test_inertia_non_negative(self, sample_3d_array):
        _, _, inertia, _ = kasba_fit(sample_3d_array, n_clusters=2)
        assert inertia >= 0.0

    def test_deterministic_with_same_seed(self, sample_3d_array):
        r1 = kasba_fit(sample_3d_array, n_clusters=2, random_seed=42)
        r2 = kasba_fit(sample_3d_array, n_clusters=2, random_seed=42)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_almost_equal(r1[1], r2[1])

    def test_different_seed_may_differ(self, sample_3d_array):
        r1 = kasba_fit(sample_3d_array, n_clusters=2, random_seed=1)
        r2 = kasba_fit(sample_3d_array, n_clusters=2, random_seed=999)
        # Labels may differ (not guaranteed, but likely with different seeds)
        # At minimum, both should be valid
        assert set(r1[0].tolist()).issubset({0, 1})
        assert set(r2[0].tolist()).issubset({0, 1})

    def test_multivariate_independent(self, multivariate_3d_array):
        labels, centers, inertia, _ = kasba_fit(multivariate_3d_array, n_clusters=3, independent=True)
        assert labels.shape == (12,)
        assert centers.shape == (3, 3 * 15)
        assert inertia >= 0.0

    def test_multivariate_dependent(self, multivariate_3d_array):
        labels, centers, inertia, _ = kasba_fit(multivariate_3d_array, n_clusters=3, independent=False)
        assert labels.shape == (12,)
        assert centers.shape == (3, 3 * 15)
        assert inertia >= 0.0

    def test_all_labels_in_range(self, sample_3d_array):
        labels, _, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        assert all(0 <= lbl < 2 for lbl in labels)

    def test_custom_parameters(self, sample_3d_array):
        labels, _, _, _ = kasba_fit(
            sample_3d_array,
            n_clusters=2,
            c=2.0,
            ba_subset_size=0.3,
            initial_step_size=0.1,
            decay_rate=0.2,
            n_iters=5,
            random_seed=7,
        )
        assert labels.shape == (10,)


class TestKASBAPredict:
    """Test kasba_predict binding."""

    def test_predict_returns_labels(self, sample_3d_array):
        _, centers, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        pred_labels = kasba_predict(centers, sample_3d_array)
        assert isinstance(pred_labels, np.ndarray)
        assert pred_labels.shape == (10,)
        assert pred_labels.dtype == np.int64

    def test_predict_matches_fit_labels(self, sample_3d_array):
        labels, centers, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        pred_labels = kasba_predict(centers, sample_3d_array)
        np.testing.assert_array_equal(labels, pred_labels)

    def test_predict_multivariate(self, multivariate_3d_array):
        labels, centers, _, _ = kasba_fit(multivariate_3d_array, n_clusters=3, independent=True)
        pred_labels = kasba_predict(centers, multivariate_3d_array, independent=True)
        np.testing.assert_array_equal(labels, pred_labels)

    def test_predict_new_data(self, sample_3d_array):
        """Predict on new data should return valid labels."""
        _, centers, _, _ = kasba_fit(sample_3d_array, n_clusters=2)
        # Create new data similar to cluster A (ascending)
        new_data = np.arange(20, dtype=np.float64).reshape(1, 1, 20)
        pred = kasba_predict(centers, new_data)
        assert pred.shape == (1,)
        assert 0 <= pred[0] < 2

    def test_predict_dependent_mode(self, multivariate_3d_array):
        labels, centers, _, _ = kasba_fit(multivariate_3d_array, n_clusters=3, independent=False)
        pred_labels = kasba_predict(centers, multivariate_3d_array, independent=False)
        np.testing.assert_array_equal(labels, pred_labels)
