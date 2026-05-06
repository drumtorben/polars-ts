"""Contrastive learning for time series clustering.

Trains a 1D CNN encoder via instance-level contrastive learning (NT-Xent)
on augmented views, then clusters the learned embeddings with k-means.

References
----------
- Cross-Domain Contrastive Learning for TS Clustering (AAAI 2024)
- Unsupervised Contrastive Learning for TS Clustering (Electronics, 2025)

"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

from polars_ts.clustering._augmentation import augment_pair
from polars_ts.clustering._contrastive_loss import TSEncoder, nt_xent_loss


class ContrastiveClusterer:
    """Contrastive learning time series clusterer.

    Learns representations via instance-level contrastive learning,
    then applies k-means on the embeddings.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    embedding_dim
        Dimension of learned embeddings.
    projection_dim
        Projection head dimension (used during training only).
    n_filters
        Base CNN filter count.
    max_epochs
        Training epochs for contrastive learning.
    lr
        Learning rate.
    batch_size
        Training batch size.
    temperature
        NT-Xent temperature parameter.
    jitter_sigma
        Jitter augmentation noise level.
    scale_sigma
        Scaling augmentation noise level.
    seed
        Random seed.
    id_col, target_col
        Column names.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        n_filters: int = 32,
        max_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        temperature: float = 0.5,
        jitter_sigma: float = 0.1,
        scale_sigma: float = 0.1,
        seed: int = 42,
        id_col: str = "unique_id",
        target_col: str = "y",
    ) -> None:
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.n_filters = n_filters
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.temperature = temperature
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.seed = seed
        self.id_col = id_col
        self.target_col = target_col
        self.labels_: pl.DataFrame | None = None
        self.embeddings_: np.ndarray | None = None
        self._encoder: TSEncoder | None = None

    def fit(self, df: pl.DataFrame) -> ContrastiveClusterer:
        """Train encoder via contrastive learning and cluster embeddings."""
        id_dtype = df[self.id_col].dtype
        ids = sorted(df[self.id_col].unique().cast(pl.String).to_list())
        n = len(ids)

        if self.n_clusters > n:
            raise ValueError(f"Cannot create {self.n_clusters} clusters from {n} time series")

        # Extract and pad series
        arrays: list[np.ndarray] = []
        for uid in ids:
            vals = df.filter(pl.col(self.id_col).cast(pl.String) == uid)[self.target_col].to_numpy().astype(np.float64)
            arrays.append(vals)

        max_len = max(a.shape[0] for a in arrays)
        X = np.zeros((n, max_len), dtype=np.float32)
        for i, a in enumerate(arrays):
            X[i, : a.shape[0]] = a

        # Normalize
        mean = X.mean()
        std = X.std() or 1.0
        X = (X - mean) / std

        # Train encoder
        torch.manual_seed(self.seed)

        encoder = TSEncoder(
            embedding_dim=self.embedding_dim,
            projection_dim=self.projection_dim,
            n_filters=self.n_filters,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        encoder.train()
        for _ in range(self.max_epochs):
            for (batch,) in loader:
                v1, v2 = augment_pair(
                    batch,
                    jitter_sigma=self.jitter_sigma,
                    scale_sigma=self.scale_sigma,
                )
                p1 = encoder.project(v1)
                p2 = encoder.project(v2)
                loss = nt_xent_loss(p1, p2, temperature=self.temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract embeddings
        encoder.eval()
        with torch.no_grad():
            embeddings = encoder(X_t).numpy()

        # K-means on embeddings
        assignments = self._kmeans(embeddings, self.n_clusters, self.seed)

        self._encoder = encoder
        self.embeddings_ = embeddings
        self.labels_ = pl.DataFrame(
            {self.id_col: ids, "cluster": assignments},
            schema={self.id_col: pl.String, "cluster": pl.Int64},
        ).with_columns(pl.col(self.id_col).cast(id_dtype))
        return self

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, seed: int, max_iter: int = 100) -> list[int]:
        """Run k-means on embedding vectors."""
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        indices = rng.choice(n, size=k, replace=False)
        centroids = X[indices].copy()

        assignments = [0] * n
        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = dists.argmin(axis=1).tolist()
            if new_assignments == assignments:
                break
            assignments = new_assignments
            # Update centroids
            for ci in range(k):
                members = X[np.array(assignments) == ci]
                if len(members) > 0:
                    centroids[ci] = members.mean(axis=0)

        return assignments


def contrastive_cluster(
    df: pl.DataFrame,
    k: int,
    max_epochs: int = 50,
    embedding_dim: int = 64,
    n_filters: int = 32,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    **kwargs: Any,
) -> pl.DataFrame:
    """Contrastive clustering convenience function.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    cc = ContrastiveClusterer(
        n_clusters=k,
        max_epochs=max_epochs,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        seed=seed,
        id_col=id_col,
        target_col=target_col,
        **kwargs,
    )
    cc.fit(df)
    assert cc.labels_ is not None
    return cc.labels_
