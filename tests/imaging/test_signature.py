import numpy as np
import polars as pl
import pytest

from polars_ts.imaging.signature import signature_features, to_signature_image


@pytest.fixture
def sample_data():
    """Return two series with distinct shapes."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 20 + ["B"] * 20,
            "y": [float(i) for i in range(20)] + [float(19 - i) for i in range(20)],
        }
    )


class TestSignatureFeatures:
    def test_output_type(self, sample_data):
        result = signature_features(sample_data, depth=2)
        assert isinstance(result, pl.DataFrame)

    def test_output_shape_depth1(self, sample_data):
        result = signature_features(sample_data, depth=1, augmentations=["time"])
        assert result.shape[0] == 2
        # depth=1, 2 channels (time + y) → 2 features
        assert result.shape[1] == 3  # unique_id + 2 sig columns

    def test_output_shape_depth2(self, sample_data):
        result = signature_features(sample_data, depth=2, augmentations=["time"])
        assert result.shape[0] == 2
        # depth=1 → 2, depth=2 → 4, total → 6 features
        assert result.shape[1] == 7  # unique_id + 6

    def test_output_shape_depth3(self, sample_data):
        result = signature_features(sample_data, depth=3, augmentations=["time"])
        assert result.shape[0] == 2
        # depth=1 → 2, depth=2 → 4, depth=3 → 8, total → 14
        assert result.shape[1] == 15  # unique_id + 14

    def test_series_ids_preserved(self, sample_data):
        result = signature_features(sample_data, depth=2)
        assert set(result["unique_id"].to_list()) == {"A", "B"}

    def test_different_series_different_signatures(self, sample_data):
        result = signature_features(sample_data, depth=2)
        sig_a = result.filter(pl.col("unique_id") == "A").drop("unique_id").to_numpy()
        sig_b = result.filter(pl.col("unique_id") == "B").drop("unique_id").to_numpy()
        assert not np.allclose(sig_a, sig_b)

    def test_no_augmentation(self, sample_data):
        result = signature_features(sample_data, depth=2, augmentations=[])
        assert result.shape[0] == 2
        # 1 channel, depth=1 → 1, depth=2 → 1, total → 2
        assert result.shape[1] == 3  # unique_id + 2

    def test_leadlag_augmentation(self, sample_data):
        result = signature_features(sample_data, depth=2, augmentations=["leadlag"])
        assert result.shape[0] == 2
        # leadlag on 1D → 2 channels, depth=1 → 2, depth=2 → 4, total → 6
        assert result.shape[1] == 7

    def test_basepoint_augmentation(self, sample_data):
        result = signature_features(sample_data, depth=1, augmentations=["time", "basepoint"])
        assert result.shape[0] == 2

    def test_unknown_augmentation_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unknown augmentation"):
            signature_features(sample_data, depth=1, augmentations=["invalid"])

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 10, "val": [float(i) for i in range(10)]})
        result = signature_features(df, depth=2, id_col="sid", target_col="val")
        assert "sid" in result.columns
        assert result.shape[0] == 1

    def test_depth4_without_iisignature_raises(self, sample_data):
        try:
            import iisignature  # noqa: F401

            pytest.skip("iisignature installed, cannot test fallback limit")
        except ImportError:
            with pytest.raises(ValueError, match="depth <= 3"):
                signature_features(sample_data, depth=4)

    def test_single_series(self):
        df = pl.DataFrame({"unique_id": ["A"] * 10, "y": [float(i) for i in range(10)]})
        result = signature_features(df, depth=2)
        assert result.shape[0] == 1


class TestToSignatureImage:
    def test_output_shape(self, sample_data):
        images = to_signature_image(sample_data, depth=2, augmentations=["time"])
        assert len(images) == 2
        # 2 channels → 2×2 matrix
        for img in images.values():
            assert img.shape == (2, 2)

    def test_depth_too_low_raises(self, sample_data):
        with pytest.raises(ValueError, match="depth must be >= 2"):
            to_signature_image(sample_data, depth=1)

    def test_series_ids(self, sample_data):
        images = to_signature_image(sample_data, depth=2)
        assert set(images.keys()) == {"A", "B"}

    def test_different_series_different_images(self, sample_data):
        images = to_signature_image(sample_data, depth=2)
        assert not np.allclose(images["A"], images["B"])

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 10, "val": [float(i) for i in range(10)]})
        images = to_signature_image(df, depth=2, id_col="sid", target_col="val")
        assert "X" in images

    def test_no_augmentation_1d(self, sample_data):
        images = to_signature_image(sample_data, depth=2, augmentations=[])
        for img in images.values():
            # 1 channel → 1×1 image
            assert img.shape == (1, 1)
