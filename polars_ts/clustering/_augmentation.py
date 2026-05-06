"""Time series augmentations for contrastive learning."""

from __future__ import annotations

import torch


def jitter(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to each time step."""
    return x + sigma * torch.randn_like(x)


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Multiply by a random per-sample scale factor."""
    factors = torch.normal(1.0, sigma, size=(x.shape[0], 1, 1), device=x.device)
    return x * factors


def window_crop(x: torch.Tensor, crop_ratio: float = 0.5) -> torch.Tensor:
    """Randomly crop a contiguous window and return it."""
    seq_len = x.shape[2]
    crop_len = max(1, int(seq_len * crop_ratio))
    start = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
    return x[:, :, start : start + crop_len]


def augment_pair(
    x: torch.Tensor,
    jitter_sigma: float = 0.1,
    scale_sigma: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two augmented views of the input batch."""
    v1 = scaling(jitter(x, sigma=jitter_sigma), sigma=scale_sigma)
    v2 = scaling(jitter(x, sigma=jitter_sigma), sigma=scale_sigma)
    return v1, v2
