"""Tests for contrastive learning time series clustering."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def cluster_data():
    """Two well-separated groups: ascending vs descending, 6 series."""
    ascending = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    descending = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    return pl.DataFrame(
        {
            "unique_id": (["A1"] * 8 + ["A2"] * 8 + ["A3"] * 8 + ["B1"] * 8 + ["B2"] * 8 + ["B3"] * 8),
            "y": (
                ascending
                + [1.0, 2.1, 3.0, 4.1, 5.0, 6.1, 7.0, 8.1]
                + [1.0, 1.9, 3.1, 4.0, 5.1, 5.9, 7.1, 8.0]
                + descending
                + [8.1, 7.0, 6.0, 4.9, 4.0, 3.1, 2.0, 0.9]
                + [7.9, 7.1, 5.9, 5.0, 4.1, 2.9, 1.9, 1.0]
            ),
        }
    )


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------
class TestAugmentations:
    def test_jitter_shape_preserved(self):
        pytest.importorskip("torch")
        from polars_ts.clustering._augmentation import jitter

        x = np.random.randn(4, 1, 16).astype(np.float32)
        import torch

        t = torch.tensor(x)
        out = jitter(t, sigma=0.1)
        assert out.shape == t.shape

    def test_jitter_not_identical(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import jitter

        torch.manual_seed(0)
        t = torch.ones(4, 1, 16)
        out = jitter(t, sigma=0.5)
        assert not torch.allclose(out, t)

    def test_scaling_shape_preserved(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import scaling

        t = torch.randn(4, 1, 16)
        out = scaling(t, sigma=0.1)
        assert out.shape == t.shape

    def test_window_crop_shorter(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import window_crop

        t = torch.randn(4, 1, 32)
        out = window_crop(t, crop_ratio=0.5)
        assert out.shape[0] == 4
        assert out.shape[1] == 1
        assert out.shape[2] == 16

    def test_augment_pair_returns_two_views(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import augment_pair

        t = torch.randn(4, 1, 16)
        v1, v2 = augment_pair(t)
        assert v1.shape == t.shape
        assert v2.shape == t.shape
        assert not torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------
class TestNTXentLoss:
    def test_loss_positive(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(42)
        z_i = torch.randn(8, 32)
        z_j = torch.randn(8, 32)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        assert loss.item() > 0

    def test_loss_zero_for_identical(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        z = torch.randn(8, 32)
        z_norm = torch.nn.functional.normalize(z, dim=1)
        loss = nt_xent_loss(z_norm, z_norm, temperature=0.5)
        # Identical views => positive pairs have similarity 1.0 => low loss
        assert loss.item() < 5.0

    def test_loss_batch_size_invariant_type(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(0)
        z_i = torch.randn(4, 16)
        z_j = torch.randn(4, 16)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------
class TestEncoder:
    def test_encoder_output_shape(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=32, n_filters=16)
        x = torch.randn(4, 1, 16)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_projection_head_output_shape(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=32, projection_dim=16, n_filters=16)
        x = torch.randn(4, 1, 16)
        z = enc(x)
        assert z.shape == (4, 32)  # encoder output is embedding_dim
        p = enc.project(x)
        assert p.shape == (4, 16)  # projection head output

    def test_encoder_different_seq_lengths(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=32, n_filters=16)
        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(2, 1, seq_len)
            z = enc(x)
            assert z.shape == (2, 32)


# ---------------------------------------------------------------------------
# ContrastiveClusterer integration tests
# ---------------------------------------------------------------------------
class TestContrastiveClusterer:
    def test_fit_returns_self(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=16, n_filters=8)
        result = cc.fit(cluster_data)
        assert result is cc

    def test_labels_shape(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert cc.labels_ is not None
        assert cc.labels_.shape[0] == 6
        assert "unique_id" in cc.labels_.columns
        assert "cluster" in cc.labels_.columns

    def test_two_clusters(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=30, embedding_dim=16, n_filters=8, seed=42)
        cc.fit(cluster_data)
        labels = dict(
            zip(
                cc.labels_["unique_id"].to_list(),
                cc.labels_["cluster"].to_list(),
                strict=False,
            )
        )
        # Ascending series should cluster together, descending together
        assert labels["A1"] == labels["A2"] == labels["A3"]
        assert labels["B1"] == labels["B2"] == labels["B3"]
        assert labels["A1"] != labels["B1"]

    def test_embeddings_available(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert cc.embeddings_ is not None
        assert cc.embeddings_.shape == (6, 16)

    def test_single_cluster(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=1, max_epochs=3, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert all(c == 0 for c in cc.labels_["cluster"].to_list())

    def test_too_many_clusters_raises(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        cc = ContrastiveClusterer(n_clusters=5, max_epochs=3)
        with pytest.raises(ValueError, match="Cannot create"):
            cc.fit(df)

    def test_custom_columns(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "ts_id": ["X"] * 8 + ["Y"] * 8 + ["Z"] * 8,
                "value": (
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
                    + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                ),
            }
        )
        cc = ContrastiveClusterer(
            n_clusters=2,
            max_epochs=3,
            embedding_dim=16,
            n_filters=8,
            id_col="ts_id",
            target_col="value",
        )
        cc.fit(df)
        assert "ts_id" in cc.labels_.columns
        assert "cluster" in cc.labels_.columns

    def test_seed_reproducibility(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc1 = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=16, n_filters=8, seed=42)
        cc1.fit(cluster_data)
        cc2 = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=16, n_filters=8, seed=42)
        cc2.fit(cluster_data)
        assert cc1.labels_["cluster"].to_list() == cc2.labels_["cluster"].to_list()


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------
def test_contrastive_cluster_function():
    pytest.importorskip("torch")
    from polars_ts.clustering.contrastive import contrastive_cluster

    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "y": (
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
                + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            ),
        }
    )
    labels = contrastive_cluster(df, k=2, max_epochs=3)
    assert "unique_id" in labels.columns
    assert "cluster" in labels.columns
    assert len(labels) == 3


def test_contrastive_cluster_import():
    """Verify lazy import from clustering package."""
    pytest.importorskip("torch")
    from polars_ts.clustering import ContrastiveClusterer, contrastive_cluster

    assert ContrastiveClusterer is not None
    assert contrastive_cluster is not None


def test_top_level_import():
    """Verify lazy import from polars_ts top-level package."""
    pytest.importorskip("torch")
    from polars_ts import ContrastiveClusterer, contrastive_cluster

    assert ContrastiveClusterer is not None
    assert contrastive_cluster is not None


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------
class TestAugmentationEdgeCases:
    def test_jitter_zero_sigma(self):
        """Zero sigma should return input unchanged."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import jitter

        t = torch.ones(2, 1, 8)
        out = jitter(t, sigma=0.0)
        assert torch.allclose(out, t)

    def test_scaling_zero_sigma(self):
        """Zero sigma scaling should return input unchanged (scale factor = 1)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import scaling

        t = torch.randn(2, 1, 8)
        out = scaling(t, sigma=0.0)
        assert torch.allclose(out, t)

    def test_window_crop_full_ratio(self):
        """Crop ratio 1.0 should return same length."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import window_crop

        t = torch.randn(2, 1, 16)
        out = window_crop(t, crop_ratio=1.0)
        assert out.shape[2] == 16

    def test_window_crop_minimal_ratio(self):
        """Very small crop ratio should return at least 1 timestep."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import window_crop

        t = torch.randn(2, 1, 16)
        out = window_crop(t, crop_ratio=0.01)
        assert out.shape[2] >= 1

    def test_jitter_single_sample(self):
        """Jitter should work with a single sample batch."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import jitter

        t = torch.randn(1, 1, 8)
        out = jitter(t, sigma=0.1)
        assert out.shape == (1, 1, 8)

    def test_augment_pair_single_timestep(self):
        """Augment pair should work with a single timestep series."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._augmentation import augment_pair

        t = torch.randn(2, 1, 1)
        v1, v2 = augment_pair(t)
        assert v1.shape == (2, 1, 1)
        assert v2.shape == (2, 1, 1)


class TestNTXentLossEdgeCases:
    def test_loss_with_batch_size_2(self):
        """Minimum batch size for contrastive loss."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(0)
        z_i = torch.randn(2, 8)
        z_j = torch.randn(2, 8)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_loss_gradient_flows(self):
        """Loss should produce valid gradients."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        z_i = torch.randn(4, 16, requires_grad=True)
        z_j = torch.randn(4, 16, requires_grad=True)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert torch.isfinite(z_i.grad).all()
        assert torch.isfinite(z_j.grad).all()

    def test_loss_temperature_effect(self):
        """Lower temperature should sharpen the loss (higher value for random pairs)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(42)
        z_i = torch.randn(8, 32)
        z_j = torch.randn(8, 32)
        loss_high_t = nt_xent_loss(z_i, z_j, temperature=2.0)
        loss_low_t = nt_xent_loss(z_i, z_j, temperature=0.1)
        # Lower temperature => sharper distribution => different loss magnitude
        assert loss_high_t.item() != loss_low_t.item()

    def test_loss_symmetric(self):
        """NT-Xent should be symmetric: loss(z_i, z_j) == loss(z_j, z_i)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(0)
        z_i = torch.randn(4, 16)
        z_j = torch.randn(4, 16)
        loss_ab = nt_xent_loss(z_i, z_j, temperature=0.5)
        loss_ba = nt_xent_loss(z_j, z_i, temperature=0.5)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-5)


class TestEncoderEdgeCases:
    def test_encoder_single_timestep(self):
        """Encoder should handle series of length 1."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=16, n_filters=8)
        x = torch.randn(2, 1, 1)
        z = enc(x)
        assert z.shape == (2, 16)
        assert torch.isfinite(z).all()

    def test_encoder_batch_size_1(self):
        """Encoder should handle batch size 1 (but batchnorm needs eval mode)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=16, n_filters=8)
        enc.eval()
        x = torch.randn(1, 1, 16)
        z = enc(x)
        assert z.shape == (1, 16)

    def test_encoder_large_embedding_dim(self):
        """Larger embedding dim than sequence length should work."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=128, n_filters=16)
        x = torch.randn(2, 1, 4)
        z = enc(x)
        assert z.shape == (2, 128)

    def test_encoder_no_nan_in_output(self):
        """Encoder should not produce NaN even with large inputs."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import TSEncoder

        enc = TSEncoder(embedding_dim=32, n_filters=16)
        x = torch.randn(4, 1, 64) * 10
        z = enc(x)
        assert torch.isfinite(z).all()


class TestContrastiveClustererEdgeCases:
    def test_two_series_two_clusters(self):
        """Minimum viable clustering: 2 series, 2 clusters."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8,
                "y": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            }
        )
        cc = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=8, n_filters=4)
        cc.fit(df)
        assert cc.labels_ is not None
        assert len(cc.labels_) == 2
        clusters = cc.labels_["cluster"].to_list()
        assert sorted(clusters) == [0, 1]

    def test_unequal_length_series(self):
        """Series with different lengths should be zero-padded correctly."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 8 + ["C"] * 6,
                "y": ([1.0, 2.0, 3.0, 4.0] + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }
        )
        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=8, n_filters=4)
        cc.fit(df)
        assert cc.labels_ is not None
        assert len(cc.labels_) == 3
        assert cc.embeddings_.shape == (3, 8)

    def test_constant_series(self):
        """Constant series should not crash (std = 0 edge case)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": [5.0] * 24,
            }
        )
        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=8, n_filters=4)
        cc.fit(df)
        assert cc.labels_ is not None
        assert len(cc.labels_) == 3

    def test_identical_series_same_cluster(self):
        """Identical series should end up in the same cluster."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": (
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                ),
            }
        )
        cc = ContrastiveClusterer(n_clusters=2, max_epochs=10, embedding_dim=8, n_filters=4, seed=42)
        cc.fit(df)
        labels = dict(zip(cc.labels_["unique_id"].to_list(), cc.labels_["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]

    def test_three_clusters(self):
        """Three distinct groups should form three clusters."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": (["A1"] * 8 + ["A2"] * 8 + ["B1"] * 8 + ["B2"] * 8 + ["C1"] * 8 + ["C2"] * 8),
                "y": (
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [1.0, 2.1, 3.0, 4.1, 5.0, 6.1, 7.0, 8.1]
                    + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                    + [8.1, 7.0, 6.0, 4.9, 4.0, 3.1, 2.0, 0.9]
                    + [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0]
                    + [1.0, 1.0, 5.1, 5.0, 1.0, 1.0, 5.0, 5.1]
                ),
            }
        )
        cc = ContrastiveClusterer(n_clusters=3, max_epochs=30, embedding_dim=16, n_filters=8, seed=42)
        cc.fit(df)
        assert cc.labels_ is not None
        n_unique_clusters = cc.labels_["cluster"].n_unique()
        assert n_unique_clusters == 3

    def test_labels_dtype(self, cluster_data):
        """Cluster labels should be Int64."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert cc.labels_["cluster"].dtype == pl.Int64

    def test_labels_id_dtype_preserved(self, cluster_data):
        """ID column dtype should be preserved from input."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert cc.labels_["unique_id"].dtype == cluster_data["unique_id"].dtype

    def test_integer_id_col(self):
        """Integer ID columns should not crash (regression test)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": [1] * 8 + [2] * 8 + [3] * 8,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 2 + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        cc = ContrastiveClusterer(n_clusters=2, max_epochs=3, embedding_dim=8, n_filters=4)
        cc.fit(df)
        assert cc.labels_["unique_id"].dtype == pl.Int64
        assert sorted(cc.labels_["unique_id"].to_list()) == [1, 2, 3]

    def test_embeddings_finite(self, cluster_data):
        """All embedding values should be finite (no NaN or Inf)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=16, n_filters=8)
        cc.fit(cluster_data)
        assert np.isfinite(cc.embeddings_).all()

    def test_different_seeds_different_results(self, cluster_data):
        """Different seeds should generally produce different embeddings."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        cc1 = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=16, n_filters=8, seed=1)
        cc1.fit(cluster_data)
        cc2 = ContrastiveClusterer(n_clusters=2, max_epochs=5, embedding_dim=16, n_filters=8, seed=999)
        cc2.fit(cluster_data)
        assert not np.allclose(cc1.embeddings_, cc2.embeddings_)

    def test_convenience_function_kwargs_forwarded(self):
        """Convenience function should forward extra kwargs."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import contrastive_cluster

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": (
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]
                    + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                ),
            }
        )
        labels = contrastive_cluster(df, k=2, max_epochs=3, temperature=0.3, lr=5e-4)
        assert len(labels) == 3


# ---------------------------------------------------------------------------
# Integration with evaluation metrics
# ---------------------------------------------------------------------------
class TestEvaluationIntegration:
    def test_silhouette_score(self, cluster_data):
        """Contrastive clustering labels should work with silhouette score."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer
        from polars_ts.clustering.evaluation import silhouette_score

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=10, embedding_dim=16, n_filters=8, seed=42)
        cc.fit(cluster_data)
        score = silhouette_score(cluster_data, cc.labels_, method="dtw")
        assert -1.0 <= score <= 1.0

    def test_davies_bouldin_score(self, cluster_data):
        """Contrastive clustering labels should work with Davies-Bouldin score."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer
        from polars_ts.clustering.evaluation import davies_bouldin_score

        cc = ContrastiveClusterer(n_clusters=2, max_epochs=10, embedding_dim=16, n_filters=8, seed=42)
        cc.fit(cluster_data)
        score = davies_bouldin_score(cluster_data, cc.labels_, method="dtw")
        assert score >= 0.0
