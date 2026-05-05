"""Contrastive loss and encoder for time series clustering."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSEncoder(nn.Module):
    """1D CNN encoder with projection head for contrastive learning.

    Parameters
    ----------
    embedding_dim
        Dimension of the learned embedding.
    projection_dim
        Dimension of the projection head output (used during training).
    n_filters
        Base number of convolutional filters.

    """

    def __init__(
        self,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        n_filters: int = 32,
    ) -> None:
        super().__init__()
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
        self.fc = nn.Linear(n_filters * 2, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding (used after training for clustering)."""
        h = self.encoder(x).squeeze(-1)
        return self.fc(h)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return projection (used during contrastive training)."""
        z = self.forward(x)
        return self.projection(z)


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Compute normalized temperature-scaled cross-entropy loss (NT-Xent).

    Parameters
    ----------
    z_i, z_j
        Projected representations of the two augmented views,
        shape ``(batch_size, projection_dim)``.
    temperature
        Temperature scaling parameter.

    """
    batch_size = z_i.shape[0]
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = torch.mm(representations, representations.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(~mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_i = torch.arange(batch_size, device=sim.device)
    pos_j = pos_i + batch_size
    labels = torch.cat([pos_j, pos_i], dim=0)

    return F.cross_entropy(sim, labels)
