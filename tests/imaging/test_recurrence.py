import numpy as np
import polars as pl
import pytest

scipy = pytest.importorskip("scipy")

from polars_ts.imaging.recurrence import rqa_features, to_recurrence_plot  # noqa: E402


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "y": [float(i) for i in range(10)] + [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


class TestToRecurrencePlot:
    def test_output_shape(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=0.5)
        assert len(images) == 2
        assert images["A"].shape == (10, 10)
        assert images["B"].shape == (10, 10)

    def test_diagonal_is_zero(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=0.5)
        for img in images.values():
            np.testing.assert_array_equal(np.diag(img), np.ones(10))

    def test_symmetric(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=0.5)
        for img in images.values():
            np.testing.assert_array_equal(img, img.T)

    def test_binary_values(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=0.5)
        for img in images.values():
            assert set(np.unique(img)).issubset({0.0, 1.0})

    def test_grayscale(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=None)
        for img in images.values():
            assert img.dtype == np.float64
            assert img.min() >= 0

    def test_no_normalize(self, sample_data):
        img_norm = to_recurrence_plot(sample_data, threshold=None, normalize=True)
        img_raw = to_recurrence_plot(sample_data, threshold=None, normalize=False)
        # Should produce different distance values
        assert not np.allclose(img_norm["A"], img_raw["A"])

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "sid": ["X"] * 5,
                "val": [1.0, 2.0, 3.0, 2.0, 1.0],
            }
        )
        images = to_recurrence_plot(df, id_col="sid", target_col="val")
        assert "X" in images
        assert images["X"].shape == (5, 5)

    def test_single_series(self):
        df = pl.DataFrame({"unique_id": ["A"] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        images = to_recurrence_plot(df)
        assert len(images) == 1


class TestRQAFeatures:
    def test_keys(self):
        R = np.eye(5)
        features = rqa_features(R)
        expected_keys = {"recurrence_rate", "determinism", "laminarity", "mean_diagonal", "mean_vertical", "entropy"}
        assert set(features.keys()) == expected_keys

    def test_identity_matrix(self):
        R = np.eye(10)
        features = rqa_features(R)
        assert features["recurrence_rate"] == pytest.approx(0.1)

    def test_full_recurrence(self):
        R = np.ones((5, 5))
        features = rqa_features(R)
        assert features["recurrence_rate"] == pytest.approx(1.0)
        assert features["determinism"] > 0

    def test_periodic_signal(self, sample_data):
        images = to_recurrence_plot(sample_data, threshold=0.5)
        features = rqa_features(images["B"])
        assert features["recurrence_rate"] > 0
