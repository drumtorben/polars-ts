"""Deep Embedded Clustering (DEC) and Improved DEC (IDEC).

Pretrains a convolutional autoencoder on reconstruction, then fine-tunes
with a KL-divergence clustering loss. IDEC additionally keeps the
reconstruction loss during fine-tuning.

References
----------
- Xie et al. (2016). *Unsupervised Deep Embedding for Clustering Analysis.* ICML.
- Guo et al. (2017). *Improved Deep Embedded Clustering with Local Structure
  Preservation.* IJCAI.
- Autoencoder-based Deep Clustering Survey (2025). arXiv:2504.02087.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from polars_ts.clustering._autoencoder import (
    ClusteringLayer,
    TSAutoencoder,
    target_distribution,
)


class DECClusterer:
    """Deep Embedded Clustering for time series.

    Two-phase training:
    1. **Pretrain** autoencoder with MSE reconstruction loss.
    2. **Fine-tune** encoder + clustering layer with KL divergence loss.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    embedding_dim
        Bottleneck embedding dimension.
    n_filters
        Base CNN filter count.
    pretrain_epochs
        Epochs for autoencoder pretraining.
    finetune_epochs
        Epochs for clustering fine-tuning.
    lr
        Learning rate.
    batch_size
        Training batch size.
    seed
        Random seed.
    id_col, target_col
        Column names.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        embedding_dim: int = 64,
        n_filters: int = 32,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
    ) -> None:
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.id_col = id_col
        self.target_col = target_col
        self.labels_: pl.DataFrame | None = None
        self.embeddings_: np.ndarray | None = None
        self._autoencoder: TSAutoencoder | None = None

    def fit(self, df: pl.DataFrame) -> DECClusterer:
        """Pretrain autoencoder, then fine-tune with clustering loss."""
        id_dtype = df[self.id_col].dtype
        ids, X_t = self._prepare_data(df)

        torch.manual_seed(self.seed)
        seq_len = X_t.shape[2]

        # Phase 1: pretrain autoencoder
        ae = TSAutoencoder(
            seq_len=seq_len,
            embedding_dim=self.embedding_dim,
            n_filters=self.n_filters,
        )
        self._pretrain(ae, X_t)

        # Initialize clustering layer centroids via k-means
        ae.eval()
        with torch.no_grad():
            z_init = ae.encode(X_t).numpy()
        centroids = self._kmeans_centroids(z_init, self.n_clusters, self.seed)

        cl = ClusteringLayer(self.n_clusters, self.embedding_dim)
        cl.centroids.data = torch.tensor(centroids, dtype=torch.float32)

        # Phase 2: fine-tune with KL divergence
        self._finetune(ae, cl, X_t)

        # Extract final embeddings and assignments
        ae.eval()
        with torch.no_grad():
            z_final = ae.encode(X_t)
            q = cl(z_final)
            assignments = q.argmax(dim=1).numpy().tolist()
            embeddings = z_final.numpy()

        self._autoencoder = ae
        self.embeddings_ = embeddings
        self.labels_ = pl.DataFrame(
            {self.id_col: ids, "cluster": assignments},
            schema={self.id_col: pl.String, "cluster": pl.Int64},
        ).with_columns(pl.col(self.id_col).cast(id_dtype))
        return self

    def _prepare_data(self, df: pl.DataFrame) -> tuple[list[str], torch.Tensor]:
        """Extract, pad, and normalize series."""
        ids = sorted(df[self.id_col].unique().cast(pl.String).to_list())
        n = len(ids)

        if self.n_clusters > n:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n} time series")

        arrays: list[np.ndarray] = []
        for uid in ids:
            vals = df.filter(pl.col(self.id_col).cast(pl.String) == uid)[self.target_col].to_numpy().astype(np.float64)
            arrays.append(vals)

        max_len = max(a.shape[0] for a in arrays)
        X = np.zeros((n, max_len), dtype=np.float32)
        for i, a in enumerate(arrays):
            X[i, : a.shape[0]] = a

        mean = X.mean()
        std = X.std() or 1.0
        X = (X - mean) / std

        return ids, torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    def _pretrain(self, ae: TSAutoencoder, X_t: torch.Tensor) -> None:
        """Pretrain autoencoder with reconstruction loss."""
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        ae.train()
        for _ in range(self.pretrain_epochs):
            for (batch,) in loader:
                _, x_hat = ae(batch)
                loss = F.mse_loss(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _finetune(
        self,
        ae: TSAutoencoder,
        cl: ClusteringLayer,
        X_t: torch.Tensor,
    ) -> None:
        """Fine-tune encoder + clustering layer with KL divergence."""
        params = list(ae.encoder.parameters()) + list(ae.fc_enc.parameters()) + list(cl.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X_t, torch.arange(X_t.shape[0]))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + 1),
        )

        ae.train()
        for _ in range(self.finetune_epochs):
            # Compute target distribution from full data
            ae.eval()
            with torch.no_grad():
                q_full = cl(ae.encode(X_t))
                p_full = target_distribution(q_full)
            ae.train()

            for batch_x, batch_idx in loader:
                z = ae.encode(batch_x)
                q = cl(z)
                p = p_full[batch_idx]
                loss = F.kl_div(q.log(), p, reduction="batchmean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @staticmethod
    def _kmeans_centroids(X: np.ndarray, k: int, seed: int, max_iter: int = 100) -> np.ndarray:
        """Run k-means and return centroids."""
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        indices = rng.choice(n, size=k, replace=False)
        centroids = X[indices].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            assignments = dists.argmin(axis=1)
            new_centroids = np.empty_like(centroids)
            for ci in range(k):
                members = X[assignments == ci]
                new_centroids[ci] = members.mean(axis=0) if len(members) > 0 else centroids[ci]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids


class IDECClusterer(DECClusterer):
    """Improved Deep Embedded Clustering for time series.

    Like DEC but keeps the reconstruction loss during fine-tuning,
    weighted by ``gamma``.

    Parameters
    ----------
    gamma
        Weight of reconstruction loss during fine-tuning. Default 0.1.

    All other parameters are inherited from :class:`DECClusterer`.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        embedding_dim: int = 64,
        n_filters: int = 32,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        gamma: float = 0.1,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            id_col=id_col,
            target_col=target_col,
        )
        self.gamma = gamma

    def _finetune(
        self,
        ae: TSAutoencoder,
        cl: ClusteringLayer,
        X_t: torch.Tensor,
    ) -> None:
        """Fine-tune with KL divergence + reconstruction loss."""
        optimizer = torch.optim.Adam(list(ae.parameters()) + list(cl.parameters()), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X_t, torch.arange(X_t.shape[0]))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + 1),
        )

        ae.train()
        for _ in range(self.finetune_epochs):
            ae.eval()
            with torch.no_grad():
                q_full = cl(ae.encode(X_t))
                p_full = target_distribution(q_full)
            ae.train()

            for batch_x, batch_idx in loader:
                z, x_hat = ae(batch_x)
                q = cl(z)
                p = p_full[batch_idx]
                kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
                recon_loss = F.mse_loss(x_hat, batch_x)
                loss = kl_loss + self.gamma * recon_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def dec_cluster(
    df: pl.DataFrame,
    k: int,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 50,
    embedding_dim: int = 64,
    n_filters: int = 32,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **kwargs: Any,
) -> pl.DataFrame:
    """DEC clustering convenience function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    dec = DECClusterer(
        n_clusters=k,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        seed=seed,
        id_col=id_col,
        target_col=target_col,
        **kwargs,
    )
    dec.fit(df)
    assert dec.labels_ is not None
    return dec.labels_


def idec_cluster(
    df: pl.DataFrame,
    k: int,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 50,
    embedding_dim: int = 64,
    n_filters: int = 32,
    gamma: float = 0.1,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **kwargs: Any,
) -> pl.DataFrame:
    """IDEC clustering convenience function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    idec = IDECClusterer(
        n_clusters=k,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        gamma=gamma,
        seed=seed,
        id_col=id_col,
        target_col=target_col,
        **kwargs,
    )
    idec.fit(df)
    assert idec.labels_ is not None
    return idec.labels_
