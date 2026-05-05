"""Autoencoder and clustering layer for deep embedded clustering."""

from __future__ import annotations

import torch
import torch.nn as nn


class TSAutoencoder(nn.Module):
    """1D convolutional autoencoder for time series.

    Parameters
    ----------
    seq_len
        Length of (padded) input sequences.
    embedding_dim
        Dimension of the bottleneck embedding.
    n_filters
        Base number of convolutional filters.

    """

    def __init__(
        self,
        seq_len: int,
        embedding_dim: int = 64,
        n_filters: int = 32,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_enc = nn.Linear(n_filters * 2, embedding_dim)

        # Decoder
        self.fc_dec = nn.Linear(embedding_dim, n_filters * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.ConvTranspose1d(n_filters, 1, kernel_size=3, padding=1, bias=False),
        )
        self._dec_upsample = nn.Upsample(size=seq_len, mode="linear", align_corners=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck embedding."""
        h = self.encoder(x).squeeze(-1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from embedding."""
        h = self.fc_dec(z).unsqueeze(-1)  # (B, C, 1)
        h = self._dec_upsample(h)  # (B, C, seq_len)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, reconstruction)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class ClusteringLayer(nn.Module):
    """Soft assignment via Student's t-distribution (DEC).

    Parameters
    ----------
    n_clusters
        Number of clusters.
    embedding_dim
        Dimension of input embeddings.
    alpha
        Degrees of freedom of the Student's t-distribution. Default 1.0.

    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, embedding_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute soft cluster assignments q_ij."""
        # (B, 1, D) - (1, K, D) => (B, K, D)
        diff = z.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        numerator = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute sharpened target distribution from soft assignments.

    Squares and normalizes q to produce a sharper auxiliary target
    that emphasizes high-confidence assignments.
    """
    weight = q**2 / q.sum(dim=0, keepdim=True)
    return weight / weight.sum(dim=1, keepdim=True)
