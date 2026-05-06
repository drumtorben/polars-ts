"""Tests for deep embedded clustering (DEC / IDEC)."""

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
# Autoencoder tests
# ---------------------------------------------------------------------------
class TestTSAutoencoder:
    def test_reconstruction_shape(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import TSAutoencoder

        ae = TSAutoencoder(seq_len=16, embedding_dim=8, n_filters=8)
        x = torch.randn(4, 1, 16)
        z, x_hat = ae(x)
        assert z.shape == (4, 8)
        assert x_hat.shape == (4, 1, 16)

    def test_encode_only(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import TSAutoencoder

        ae = TSAutoencoder(seq_len=16, embedding_dim=8, n_filters=8)
        x = torch.randn(4, 1, 16)
        z = ae.encode(x)
        assert z.shape == (4, 8)

    def test_different_seq_lengths(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import TSAutoencoder

        for seq_len in [8, 16, 32]:
            ae = TSAutoencoder(seq_len=seq_len, embedding_dim=8, n_filters=8)
            x = torch.randn(2, 1, seq_len)
            z, x_hat = ae(x)
            assert z.shape == (2, 8)
            assert x_hat.shape == (2, 1, seq_len)

    def test_reconstruction_loss_decreases(self):
        """Autoencoder should learn to reconstruct after a few epochs."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import TSAutoencoder

        torch.manual_seed(42)
        ae = TSAutoencoder(seq_len=16, embedding_dim=8, n_filters=8)
        x = torch.randn(8, 1, 16)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

        ae.train()
        _, x_hat0 = ae(x)
        loss0 = torch.nn.functional.mse_loss(x_hat0, x).item()

        for _ in range(50):
            _, x_hat = ae(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, x_hat_final = ae(x)
        loss_final = torch.nn.functional.mse_loss(x_hat_final, x).item()
        assert loss_final < loss0


# ---------------------------------------------------------------------------
# Clustering layer tests
# ---------------------------------------------------------------------------
class TestClusteringLayer:
    def test_output_shape(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer

        cl = ClusteringLayer(n_clusters=3, embedding_dim=8)
        z = torch.randn(4, 8)
        q = cl(z)
        assert q.shape == (4, 3)

    def test_output_sums_to_one(self):
        """Student-t distribution should produce valid soft assignments."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer

        cl = ClusteringLayer(n_clusters=3, embedding_dim=8)
        z = torch.randn(4, 8)
        q = cl(z)
        sums = q.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_all_values_positive(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer

        cl = ClusteringLayer(n_clusters=3, embedding_dim=8)
        z = torch.randn(4, 8)
        q = cl(z)
        assert (q > 0).all()

    def test_centroids_initialized(self):
        pytest.importorskip("torch")

        from polars_ts.clustering._autoencoder import ClusteringLayer

        cl = ClusteringLayer(n_clusters=3, embedding_dim=8)
        assert cl.centroids.shape == (3, 8)


# ---------------------------------------------------------------------------
# Target distribution tests
# ---------------------------------------------------------------------------
class TestTargetDistribution:
    def test_output_shape(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import target_distribution

        q = torch.softmax(torch.randn(4, 3), dim=1)
        p = target_distribution(q)
        assert p.shape == (4, 3)

    def test_output_sums_to_one(self):
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import target_distribution

        q = torch.softmax(torch.randn(8, 5), dim=1)
        p = target_distribution(q)
        sums = p.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_sharpens_distribution(self):
        """Target distribution should be sharper (higher entropy reduction)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import target_distribution

        q = torch.softmax(torch.randn(4, 3), dim=1)
        p = target_distribution(q)
        # p should have higher max values (sharper) than q on average
        assert p.max(dim=1).values.mean() >= q.max(dim=1).values.mean()


# ---------------------------------------------------------------------------
# DECClusterer integration tests
# ---------------------------------------------------------------------------
class TestDECClusterer:
    def test_fit_returns_self(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        result = dec.fit(cluster_data)
        assert result is dec

    def test_labels_shape(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        dec.fit(cluster_data)
        assert dec.labels_ is not None
        assert dec.labels_.shape[0] == 6
        assert "unique_id" in dec.labels_.columns
        assert "cluster" in dec.labels_.columns

    def test_two_clusters(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=20,
            finetune_epochs=20,
            embedding_dim=16,
            n_filters=8,
            seed=42,
        )
        dec.fit(cluster_data)
        labels = dict(
            zip(
                dec.labels_["unique_id"].to_list(),
                dec.labels_["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A1"] == labels["A2"] == labels["A3"]
        assert labels["B1"] == labels["B2"] == labels["B3"]
        assert labels["A1"] != labels["B1"]

    def test_embeddings_available(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        dec.fit(cluster_data)
        assert dec.embeddings_ is not None
        assert dec.embeddings_.shape == (6, 8)

    def test_single_cluster(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=1,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        dec.fit(cluster_data)
        assert all(c == 0 for c in dec.labels_["cluster"].to_list())

    def test_too_many_clusters_raises(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        dec = DECClusterer(n_clusters=5, pretrain_epochs=3, finetune_epochs=3)
        with pytest.raises(ValueError, match="Cannot create"):
            dec.fit(df)

    def test_custom_columns(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

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
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
            id_col="ts_id",
            target_col="value",
        )
        dec.fit(df)
        assert "ts_id" in dec.labels_.columns
        assert "cluster" in dec.labels_.columns

    def test_seed_reproducibility(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        kwargs = dict(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=5,
            embedding_dim=8,
            n_filters=8,
            seed=42,
        )
        dec1 = DECClusterer(**kwargs)
        dec1.fit(cluster_data)
        dec2 = DECClusterer(**kwargs)
        dec2.fit(cluster_data)
        assert dec1.labels_["cluster"].to_list() == dec2.labels_["cluster"].to_list()


# ---------------------------------------------------------------------------
# IDECClusterer tests
# ---------------------------------------------------------------------------
class TestIDECClusterer:
    def test_fit_returns_self(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import IDECClusterer

        idec = IDECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        result = idec.fit(cluster_data)
        assert result is idec

    def test_labels_shape(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import IDECClusterer

        idec = IDECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        idec.fit(cluster_data)
        assert idec.labels_ is not None
        assert idec.labels_.shape[0] == 6
        assert "unique_id" in idec.labels_.columns
        assert "cluster" in idec.labels_.columns

    def test_two_clusters(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import IDECClusterer

        idec = IDECClusterer(
            n_clusters=2,
            pretrain_epochs=20,
            finetune_epochs=20,
            embedding_dim=16,
            n_filters=8,
            seed=42,
        )
        idec.fit(cluster_data)
        labels = dict(
            zip(
                idec.labels_["unique_id"].to_list(),
                idec.labels_["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A1"] == labels["A2"] == labels["A3"]
        assert labels["B1"] == labels["B2"] == labels["B3"]
        assert labels["A1"] != labels["B1"]

    def test_reconstruction_loss_weight(self, cluster_data):
        """IDEC with gamma=0 should behave like DEC (no reconstruction)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import IDECClusterer

        idec = IDECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
            gamma=0.0,
        )
        idec.fit(cluster_data)
        assert idec.labels_ is not None

    def test_embeddings_available(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import IDECClusterer

        idec = IDECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        idec.fit(cluster_data)
        assert idec.embeddings_ is not None
        assert idec.embeddings_.shape == (6, 8)


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------
def test_dec_cluster_function():
    pytest.importorskip("torch")
    from polars_ts.clustering.deep_cluster import dec_cluster

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
    labels = dec_cluster(df, k=2, pretrain_epochs=3, finetune_epochs=3)
    assert "unique_id" in labels.columns
    assert "cluster" in labels.columns
    assert len(labels) == 3


def test_idec_cluster_function():
    pytest.importorskip("torch")
    from polars_ts.clustering.deep_cluster import idec_cluster

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
    labels = idec_cluster(df, k=2, pretrain_epochs=3, finetune_epochs=3)
    assert "unique_id" in labels.columns
    assert "cluster" in labels.columns
    assert len(labels) == 3


def test_deep_cluster_import():
    """Verify lazy import from clustering package."""
    pytest.importorskip("torch")
    from polars_ts.clustering import DECClusterer, IDECClusterer, dec_cluster, idec_cluster

    assert DECClusterer is not None
    assert IDECClusterer is not None
    assert dec_cluster is not None
    assert idec_cluster is not None


def test_top_level_import():
    """Verify lazy import from polars_ts top-level package."""
    pytest.importorskip("torch")
    from polars_ts import DECClusterer, IDECClusterer

    assert DECClusterer is not None
    assert IDECClusterer is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestDeepClusterEdgeCases:
    def test_two_series_two_clusters(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8,
                "y": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            }
        )
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=4,
        )
        dec.fit(df)
        assert len(dec.labels_) == 2
        assert sorted(dec.labels_["cluster"].to_list()) == [0, 1]

    def test_unequal_length_series(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 8 + ["C"] * 6,
                "y": ([1.0, 2.0, 3.0, 4.0] + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            }
        )
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=4,
        )
        dec.fit(df)
        assert len(dec.labels_) == 3
        assert dec.embeddings_.shape == (3, 8)

    def test_constant_series(self):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "y": [5.0] * 24,
            }
        )
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=4,
        )
        dec.fit(df)
        assert len(dec.labels_) == 3

    def test_labels_dtype(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        dec.fit(cluster_data)
        assert dec.labels_["cluster"].dtype == pl.Int64

    def test_embeddings_finite(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=5,
            finetune_epochs=3,
            embedding_dim=8,
            n_filters=8,
        )
        dec.fit(cluster_data)
        assert np.isfinite(dec.embeddings_).all()

    def test_three_clusters(self):
        """Three distinct groups with 3 series each should form 3 clusters."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        # Group A: ascending, Group B: descending, Group C: oscillating
        # 3 series per group for more statistical signal
        df = pl.DataFrame(
            {
                "unique_id": (
                    ["A1"] * 8
                    + ["A2"] * 8
                    + ["A3"] * 8
                    + ["B1"] * 8
                    + ["B2"] * 8
                    + ["B3"] * 8
                    + ["C1"] * 8
                    + ["C2"] * 8
                    + ["C3"] * 8
                ),
                "y": (
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    + [1.0, 2.1, 3.0, 4.1, 5.0, 6.1, 7.0, 8.1]
                    + [1.1, 2.0, 2.9, 4.0, 5.1, 6.0, 7.1, 8.0]
                    + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                    + [8.1, 7.0, 6.0, 4.9, 4.0, 3.1, 2.0, 0.9]
                    + [7.9, 7.1, 5.9, 5.0, 4.1, 2.9, 1.9, 1.0]
                    + [0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0]
                    + [0.0, 8.1, 0.0, 8.0, 0.1, 8.0, 0.0, 8.1]
                    + [0.1, 7.9, 0.0, 8.1, 0.0, 8.0, 0.1, 8.0]
                ),
            }
        )
        dec = DECClusterer(
            n_clusters=3,
            pretrain_epochs=50,
            finetune_epochs=50,
            embedding_dim=16,
            n_filters=8,
            seed=42,
        )
        dec.fit(df)
        assert dec.labels_["cluster"].n_unique() == 3


# ---------------------------------------------------------------------------
# Integration with evaluation metrics
# ---------------------------------------------------------------------------
class TestDeepClusterEvaluation:
    def test_silhouette_score(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer
        from polars_ts.clustering.evaluation import silhouette_score

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=10,
            finetune_epochs=5,
            embedding_dim=8,
            n_filters=8,
            seed=42,
        )
        dec.fit(cluster_data)
        score = silhouette_score(cluster_data, dec.labels_, method="dtw")
        assert -1.0 <= score <= 1.0

    def test_davies_bouldin_score(self, cluster_data):
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer
        from polars_ts.clustering.evaluation import davies_bouldin_score

        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=10,
            finetune_epochs=5,
            embedding_dim=8,
            n_filters=8,
            seed=42,
        )
        dec.fit(cluster_data)
        score = davies_bouldin_score(cluster_data, dec.labels_, method="dtw")
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Adversarial / devil's advocate tests
# ---------------------------------------------------------------------------
class TestAdversarial:
    def test_extreme_values(self):
        """Very large input values should not produce NaN (normalization handles it)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8,
                "y": ([1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6] + [-8e6, -7e6, -6e6, -5e6, -4e6, -3e6, -2e6, -1e6]),
            }
        )
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=4,
            n_filters=4,
        )
        dec.fit(df)
        assert np.isfinite(dec.embeddings_).all()
        assert dec.labels_ is not None

    def test_very_short_series(self):
        """Series of length 2 should work."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 2 + ["B"] * 2 + ["C"] * 2,
                "y": [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
            }
        )
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=4,
            n_filters=4,
        )
        dec.fit(df)
        assert len(dec.labels_) == 3

    def test_single_series_single_cluster(self):
        """One series with one cluster should work (degenerate case)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

        df = pl.DataFrame({"unique_id": ["A"] * 8, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        dec = DECClusterer(
            n_clusters=1,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=4,
            n_filters=4,
        )
        dec.fit(df)
        assert dec.labels_["cluster"].to_list() == [0]

    def test_dec_decoder_frozen_during_finetune(self):
        """DEC should NOT update decoder weights during fine-tuning."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer, TSAutoencoder, target_distribution

        torch.manual_seed(42)
        ae = TSAutoencoder(seq_len=8, embedding_dim=4, n_filters=4)
        cl = ClusteringLayer(2, 4)
        X = torch.randn(4, 1, 8)

        dec_weight_before = ae.fc_dec.weight.data.clone()

        # Simulate DEC finetune
        params = list(ae.encoder.parameters()) + list(ae.fc_enc.parameters()) + list(cl.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-2)
        ae.train()
        for _ in range(5):
            ae.eval()
            with torch.no_grad():
                q_full = cl(ae.encode(X))
                p_full = target_distribution(q_full)
            ae.train()
            z = ae.encode(X)
            q = cl(z)
            loss = torch.nn.functional.kl_div(q.log(), p_full, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert torch.allclose(dec_weight_before, ae.fc_dec.weight.data)

    def test_idec_decoder_updates_during_finetune(self):
        """IDEC should update decoder weights during fine-tuning."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer, TSAutoencoder, target_distribution

        torch.manual_seed(42)
        ae = TSAutoencoder(seq_len=8, embedding_dim=4, n_filters=4)
        cl = ClusteringLayer(2, 4)
        X = torch.randn(4, 1, 8)

        dec_weight_before = ae.fc_dec.weight.data.clone()

        optimizer = torch.optim.Adam(list(ae.parameters()) + list(cl.parameters()), lr=1e-2)
        ae.train()
        for _ in range(5):
            ae.eval()
            with torch.no_grad():
                q_full = cl(ae.encode(X))
                p_full = target_distribution(q_full)
            ae.train()
            z, x_hat = ae(X)
            q = cl(z)
            kl = torch.nn.functional.kl_div(q.log(), p_full, reduction="batchmean")
            recon = torch.nn.functional.mse_loss(x_hat, X)
            loss = kl + 0.1 * recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert not torch.allclose(dec_weight_before, ae.fc_dec.weight.data)

    def test_clustering_layer_never_zero(self):
        """Student-t soft assignment should never produce exact zero."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer

        cl = ClusteringLayer(3, 8)
        z = torch.randn(50, 8) * 100
        q = cl(z)
        assert (q > 0).all()
        assert torch.isfinite(q.log()).all()

    def test_target_distribution_sharpens_uniform(self):
        """Target distribution of a uniform q should remain uniform."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import target_distribution

        q = torch.ones(4, 3) / 3.0
        p = target_distribution(q)
        assert torch.allclose(p, q, atol=1e-5)

    def test_target_distribution_sharpens_skewed(self):
        """Target distribution should amplify confident assignments."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import target_distribution

        q = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        p = target_distribution(q)
        # p should be sharper: max should be higher
        assert p[0, 0] > q[0, 0]
        assert p[1, 1] > q[1, 1]

    def test_kl_grad_finite(self):
        """KL divergence gradients should be finite during training."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import ClusteringLayer, target_distribution

        torch.manual_seed(0)
        cl = ClusteringLayer(3, 8)
        z = torch.randn(6, 8, requires_grad=True)
        q = cl(z)
        p = target_distribution(q.detach())
        loss = torch.nn.functional.kl_div(q.log(), p, reduction="batchmean")
        loss.backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_autoencoder_reconstruction_improves(self):
        """Pretraining should reduce reconstruction error."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._autoencoder import TSAutoencoder

        torch.manual_seed(42)
        ae = TSAutoencoder(seq_len=16, embedding_dim=8, n_filters=8)
        X = torch.randn(8, 1, 16)
        optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

        ae.eval()
        with torch.no_grad():
            _, x_hat_before = ae(X)
            mse_before = torch.nn.functional.mse_loss(x_hat_before, X).item()

        ae.train()
        for _ in range(100):
            _, x_hat = ae(X)
            loss = torch.nn.functional.mse_loss(x_hat, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ae.eval()
        with torch.no_grad():
            _, x_hat_after = ae(X)
            mse_after = torch.nn.functional.mse_loss(x_hat_after, X).item()

        assert mse_after < mse_before  # reconstruction should improve

    def test_batch_size_larger_than_data(self):
        """Batch size > n_samples should work (DataLoader returns 1 batch)."""
        pytest.importorskip("torch")
        from polars_ts.clustering.deep_cluster import DECClusterer

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
        dec = DECClusterer(
            n_clusters=2,
            pretrain_epochs=3,
            finetune_epochs=3,
            embedding_dim=4,
            n_filters=4,
            batch_size=256,
        )
        dec.fit(df)
        assert len(dec.labels_) == 3

    def test_contrastive_extreme_values(self):
        """Contrastive clusterer should handle extreme input values."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 8 + ["B"] * 8,
                "y": ([1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6] + [-8e6, -7e6, -6e6, -5e6, -4e6, -3e6, -2e6, -1e6]),
            }
        )
        cc = ContrastiveClusterer(
            n_clusters=2,
            max_epochs=3,
            embedding_dim=4,
            n_filters=4,
        )
        cc.fit(df)
        assert np.isfinite(cc.embeddings_).all()

    def test_contrastive_single_series(self):
        """One series with one cluster should work for contrastive."""
        pytest.importorskip("torch")
        from polars_ts.clustering.contrastive import ContrastiveClusterer

        df = pl.DataFrame({"unique_id": ["A"] * 8, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        cc = ContrastiveClusterer(
            n_clusters=1,
            max_epochs=3,
            embedding_dim=4,
            n_filters=4,
        )
        cc.fit(df)
        assert cc.labels_["cluster"].to_list() == [0]

    def test_nt_xent_batch_size_2(self):
        """NT-Xent with minimum viable batch (2 samples, 2 negatives)."""
        pytest.importorskip("torch")
        import torch

        from polars_ts.clustering._contrastive_loss import nt_xent_loss

        torch.manual_seed(0)
        z_i = torch.randn(2, 8, requires_grad=True)
        z_j = torch.randn(2, 8, requires_grad=True)
        loss = nt_xent_loss(z_i, z_j, temperature=0.5)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(z_i.grad).all()
        assert torch.isfinite(z_j.grad).all()
