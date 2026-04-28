import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from polars_ts.imaging.embeddings import extract_vision_embeddings  # noqa: E402


@pytest.fixture
def sample_images():
    """Return 3 grayscale 32x32 sample images."""
    rng = np.random.default_rng(42)
    return {
        "A": rng.random((32, 32)).astype(np.float64),
        "B": rng.random((32, 32)).astype(np.float64),
        "C": rng.random((32, 32)).astype(np.float64),
    }


class TestExtractVisionEmbeddings:
    def test_resnet18_output_shape(self, sample_images):
        result = extract_vision_embeddings(sample_images, model="resnet18")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3
        assert "unique_id" in result.columns
        assert "emb_0" in result.columns
        # ResNet18 avgpool produces 512-d features
        assert result.shape[1] == 513  # unique_id + 512 emb columns

    def test_resnet50_output_shape(self, sample_images):
        result = extract_vision_embeddings(sample_images, model="resnet50")
        assert result.shape[0] == 3
        # ResNet50 avgpool produces 2048-d features
        assert result.shape[1] == 2049

    def test_vit_output_shape(self, sample_images):
        result = extract_vision_embeddings(sample_images, model="vit_b_16")
        assert result.shape[0] == 3
        assert "unique_id" in result.columns
        # ViT-B/16 CLS token is 768-d
        assert result.shape[1] == 769

    def test_series_ids_preserved(self, sample_images):
        result = extract_vision_embeddings(sample_images, model="resnet18")
        ids = result["unique_id"].to_list()
        assert set(ids) == {"A", "B", "C"}

    def test_deterministic(self, sample_images):
        r1 = extract_vision_embeddings(sample_images, model="resnet18")
        r2 = extract_vision_embeddings(sample_images, model="resnet18")
        np.testing.assert_array_equal(
            r1.drop("unique_id").to_numpy(),
            r2.drop("unique_id").to_numpy(),
        )

    def test_single_image(self):
        images = {"X": np.random.default_rng(0).random((16, 16))}
        result = extract_vision_embeddings(images, model="resnet18")
        assert result.shape[0] == 1
        assert result["unique_id"][0] == "X"

    def test_batch_size(self, sample_images):
        r1 = extract_vision_embeddings(sample_images, model="resnet18", batch_size=1)
        r2 = extract_vision_embeddings(sample_images, model="resnet18", batch_size=32)
        np.testing.assert_allclose(
            r1.sort("unique_id").drop("unique_id").to_numpy(),
            r2.sort("unique_id").drop("unique_id").to_numpy(),
            atol=1e-5,
        )

    def test_unknown_model_raises(self, sample_images):
        with pytest.raises(ValueError, match="Unknown model"):
            extract_vision_embeddings(sample_images, model="nonexistent")

    def test_constant_image(self):
        images = {"A": np.ones((20, 20))}
        result = extract_vision_embeddings(images, model="resnet18")
        assert result.shape[0] == 1
        assert not np.any(np.isnan(result.drop("unique_id").to_numpy()))

    def test_different_image_sizes(self):
        """Images of different sizes should all be resized to 224x224."""
        images = {
            "small": np.random.default_rng(0).random((10, 10)),
            "large": np.random.default_rng(1).random((500, 500)),
        }
        result = extract_vision_embeddings(images, model="resnet18")
        assert result.shape[0] == 2
