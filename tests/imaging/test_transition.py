import numpy as np
import polars as pl
import pytest

from polars_ts.imaging.transition import to_mtf


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 20,
            "y": [float(i % 5) for i in range(20)],
        }
    )


class TestMTF:
    def test_output_shape(self, sample_data):
        images = to_mtf(sample_data)
        assert len(images) == 1
        assert images["A"].shape == (20, 20)

    def test_values_in_range(self, sample_data):
        images = to_mtf(sample_data)
        for img in images.values():
            assert img.min() >= -1e-10
            assert img.max() <= 1.0 + 1e-10

    def test_n_bins_parameter(self, sample_data):
        img_4 = to_mtf(sample_data, n_bins=4)["A"]
        img_8 = to_mtf(sample_data, n_bins=8)["A"]
        # Different binning should produce different images
        assert not np.allclose(img_4, img_8)

    def test_image_size_downsampling(self, sample_data):
        images = to_mtf(sample_data, image_size=5)
        assert images["A"].shape == (5, 5)

    def test_image_size_none_full_resolution(self, sample_data):
        images = to_mtf(sample_data, image_size=None)
        assert images["A"].shape == (20, 20)

    def test_multi_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 10 + ["B"] * 10,
                "y": list(range(10)) + list(range(10)),
            }
        )
        images = to_mtf(df)
        assert len(images) == 2
        assert images["A"].shape == (10, 10)
        assert images["B"].shape == (10, 10)

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 10, "val": [float(i) for i in range(10)]})
        images = to_mtf(df, id_col="sid", target_col="val")
        assert "X" in images
