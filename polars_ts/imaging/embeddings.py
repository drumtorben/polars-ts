"""Vision model embedding extraction for time series images.

Extracts feature embeddings from pretrained vision models (ResNet, ViT, CLIP)
applied to time series image representations (recurrence plots, GAF, MTF, etc.).

Requires ``torch`` and ``torchvision`` (for ResNet/ViT) or ``transformers``
(for CLIP).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _ensure_torch() -> Any:
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required for vision embeddings. Install with: pip install torch torchvision"
        ) from None
    return torch


def _images_to_tensor(
    images: dict[str, np.ndarray],
    target_size: tuple[int, int] = (224, 224),
) -> tuple[list[str], Any]:
    """Convert dict of 2D arrays to a batch tensor suitable for vision models.

    Grayscale images are duplicated to 3 channels. All images are resized
    to ``target_size`` and normalised with ImageNet statistics.
    """
    torch = _ensure_torch()
    import torch.nn.functional as F

    ids = list(images.keys())
    tensors = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for sid in ids:
        img = images[sid].astype(np.float32)
        # Normalise image values to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax - vmin > 0:
            img = (img - vmin) / (vmax - vmin)
        # Convert to tensor: (1, H, W)
        t = torch.from_numpy(img).unsqueeze(0)
        # Resize to target_size
        t = F.interpolate(t.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
        # Duplicate to 3 channels
        t = t.expand(3, -1, -1)
        # ImageNet normalisation
        t = (t - mean) / std
        tensors.append(t)

    batch = torch.stack(tensors)
    return ids, batch


def _get_torchvision_model(model_name: str) -> tuple[Any, str]:
    """Load a pretrained torchvision model and return (model, layer_name)."""
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError(
            "torchvision is required for ResNet/ViT embeddings. Install with: pip install torchvision"
        ) from None

    weights_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "avgpool"),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "avgpool"),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, "encoder"),
    }

    if model_name not in weights_map:
        raise ValueError(f"Unknown model {model_name!r}. Supported: {list(weights_map.keys())}")

    factory, weights, default_layer = weights_map[model_name]
    model = factory(weights=weights)
    model.eval()
    return model, default_layer


def _extract_resnet(model: Any, batch: Any) -> np.ndarray:
    """Extract avgpool features from a ResNet model."""
    torch = _ensure_torch()
    activations: list[Any] = []

    def hook(_module: Any, _input: Any, output: Any) -> None:
        activations.append(output)

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(batch)
    handle.remove()

    features = activations[0].squeeze(-1).squeeze(-1)
    return features.cpu().numpy()


def _extract_vit(model: Any, batch: Any) -> np.ndarray:
    """Extract CLS token features from a ViT model.

    Replaces ``model.heads`` with an identity to get pre-head features,
    then restores it. This avoids relying on private ViT internals.
    """
    torch = _ensure_torch()

    original_heads = model.heads
    model.heads = torch.nn.Identity()
    try:
        with torch.no_grad():
            features = model(batch)
    finally:
        model.heads = original_heads

    return features.cpu().numpy()


def _extract_clip(
    images: dict[str, np.ndarray],
    model_name: str,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    """Extract CLIP vision embeddings."""
    _ensure_torch()
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers is required for CLIP embeddings. Install with: pip install transformers"
        ) from None

    import torch

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for CLIP embeddings. Install with: pip install Pillow") from None

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    ids = list(images.keys())
    all_embeddings = []

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        pil_images = []
        for sid in batch_ids:
            img = images[sid].astype(np.float32)
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > 0:
                img = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
            pil_images.append(Image.fromarray(img, mode="L").convert("RGB"))

        inputs = processor(images=pil_images, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        all_embeddings.append(outputs.cpu().numpy())

    return ids, np.concatenate(all_embeddings, axis=0)


def extract_vision_embeddings(
    images: dict[str, np.ndarray],
    model: str = "resnet18",
    batch_size: int = 32,
) -> pl.DataFrame:
    """Extract feature embeddings from a pretrained vision model.

    Takes time series images (from ``to_recurrence_plot``, ``to_gasf``,
    ``to_gadf``, or ``to_mtf``) and passes them through a pretrained
    vision model to extract feature vectors.

    Parameters
    ----------
    images
        Mapping from series ID to a 2D numpy array (image).
        Typically the output of ``to_gasf``, ``to_recurrence_plot``, etc.
    model
        Model name. Supported: ``"resnet18"``, ``"resnet50"``,
        ``"vit_b_16"``, ``"clip"`` (uses ``openai/clip-vit-base-patch32``).
    batch_size
        Number of images to process at once (for ResNet/ViT).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``["unique_id", "emb_0", "emb_1", ..., "emb_d"]``.

    """
    if model == "clip":
        ids, embeddings = _extract_clip(images, "openai/clip-vit-base-patch32", batch_size)
    else:
        _ensure_torch()
        net, _ = _get_torchvision_model(model)

        is_vit = model.startswith("vit")
        extract_fn = _extract_vit if is_vit else _extract_resnet

        all_ids: list[str] = []
        all_features: list[np.ndarray] = []

        keys = list(images.keys())
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i : i + batch_size]
            batch_images = {k: images[k] for k in batch_keys}
            batch_ids, batch_tensor = _images_to_tensor(batch_images)

            features = extract_fn(net, batch_tensor)
            all_ids.extend(batch_ids)
            all_features.append(features)

        ids = all_ids
        embeddings = np.concatenate(all_features, axis=0)

    n_dim = embeddings.shape[1]
    data: dict[str, Any] = {"unique_id": ids}
    for i in range(n_dim):
        data[f"emb_{i}"] = embeddings[:, i].tolist()

    return pl.DataFrame(data)
