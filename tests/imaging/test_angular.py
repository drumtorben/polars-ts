import numpy as np
import polars as pl
import pytest

from polars_ts.imaging.angular import to_gadf, to_gasf


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10 + ["B"] * 10,
            "y": [float(i) for i in range(10)] + [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


class TestGASF:
    def test_output_shape(self, sample_data):
        images = to_gasf(sample_data)
        assert len(images) == 2
        assert images["A"].shape == (10, 10)

    def test_values_in_range(self, sample_data):
        images = to_gasf(sample_data)
        for img in images.values():
            assert img.min() >= -1.0 - 1e-10
            assert img.max() <= 1.0 + 1e-10

    def test_symmetric(self, sample_data):
        images = to_gasf(sample_data)
        for img in images.values():
            np.testing.assert_allclose(img, img.T, atol=1e-10)

    def test_image_size_downsampling(self, sample_data):
        images = to_gasf(sample_data, image_size=5)
        for img in images.values():
            assert img.shape == (5, 5)

    def test_image_size_none_full_resolution(self, sample_data):
        images = to_gasf(sample_data, image_size=None)
        assert images["A"].shape == (10, 10)

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 5, "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        images = to_gasf(df, id_col="sid", target_col="val")
        assert "X" in images

    def test_constant_series(self):
        df = pl.DataFrame({"unique_id": ["A"] * 5, "y": [1.0] * 5})
        images = to_gasf(df)
        assert images["A"].shape == (5, 5)


class TestGADF:
    def test_output_shape(self, sample_data):
        images = to_gadf(sample_data)
        assert len(images) == 2
        assert images["A"].shape == (10, 10)

    def test_values_in_range(self, sample_data):
        images = to_gadf(sample_data)
        for img in images.values():
            assert img.min() >= -1.0 - 1e-10
            assert img.max() <= 1.0 + 1e-10

    def test_antisymmetric(self, sample_data):
        """GADF = sin(phi_i - phi_j) should be antisymmetric."""
        images = to_gadf(sample_data)
        for img in images.values():
            np.testing.assert_allclose(img, -img.T, atol=1e-10)

    def test_diagonal_is_zero(self, sample_data):
        """sin(phi_i - phi_i) = 0."""
        images = to_gadf(sample_data)
        for img in images.values():
            np.testing.assert_allclose(np.diag(img), 0, atol=1e-10)

    def test_image_size_downsampling(self, sample_data):
        images = to_gadf(sample_data, image_size=4)
        for img in images.values():
            assert img.shape == (4, 4)
