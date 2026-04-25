"""Foundation model embedding adapters (Chronos, MOMENT).

Extract fixed-length embedding vectors from pre-trained time series
foundation models, returning a polars DataFrame suitable for downstream
clustering or classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _extract_series(
    df: pl.DataFrame,
    target_col: str,
    id_col: str,
    time_col: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Extract per-series arrays, preserving variable lengths."""
    sort_cols = [id_col, time_col] if time_col in df.columns else [id_col]
    sorted_df = df.sort(sort_cols)
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    for key, group in sorted_df.group_by(id_col, maintain_order=True):
        ids.append(str(key[0] if isinstance(key, tuple) else key))
        arrays.append(group[target_col].to_numpy().astype(np.float32))
    return ids, arrays


def _arrays_to_result(
    ids: list[str],
    embeddings: np.ndarray,
    id_col: str,
    prefix: str,
) -> pl.DataFrame:
    """Convert id list + embedding matrix to a polars DataFrame."""
    n_dim = embeddings.shape[1]
    data: dict[str, Any] = {id_col: ids}
    for i in range(n_dim):
        data[f"{prefix}_{i}"] = embeddings[:, i].tolist()
    return pl.DataFrame(data)


def to_chronos_embeddings(
    df: pl.DataFrame,
    model_name: str = "amazon/chronos-t5-small",
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    device: str = "cpu",
    batch_size: int = 32,
) -> pl.DataFrame:
    """Extract embeddings from a Chronos foundation model.

    Loads the specified Chronos model and extracts encoder embeddings
    for each time series, mean-pooled over the time dimension.

    Requires ``torch`` and ``transformers``.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    model_name
        HuggingFace model identifier for a Chronos model.
    target_col
        Column with the values to embed.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    device
        Torch device (``"cpu"``, ``"cuda"``, etc.).
    batch_size
        Number of series to process at once.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, emb_0, emb_1, ..., emb_d]``.

    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for Chronos embeddings. Install with: pip install torch") from None

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for Chronos embeddings. Install with: pip install transformers"
        ) from None

    ids, arrays = _extract_series(df, target_col, id_col, time_col)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(arrays), batch_size):
        batch = arrays[start : start + batch_size]
        # Tokenize: Chronos tokenizer expects list of 1-D tensors
        inputs = tokenizer(
            [torch.tensor(a, dtype=torch.float32) for a in batch],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean-pool over sequence dimension
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        pooled = hidden.mean(dim=1).cpu().numpy()  # (batch, hidden_dim)
        all_embeddings.append(pooled)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return _arrays_to_result(ids, embeddings, id_col, "emb")


def to_moment_embeddings(
    df: pl.DataFrame,
    model_name: str = "AutonLab/MOMENT-1-large",
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    device: str = "cpu",
) -> pl.DataFrame:
    """Extract embeddings from a MOMENT foundation model.

    Loads the specified MOMENT model and extracts representation
    embeddings for each time series.

    Requires ``torch`` and ``momentfm``.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    model_name
        HuggingFace model identifier for a MOMENT model.
    target_col
        Column with the values to embed.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    device
        Torch device (``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, emb_0, emb_1, ..., emb_d]``.

    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for MOMENT embeddings. Install with: pip install torch") from None

    try:
        from momentfm import MOMENTPipeline
    except ImportError:
        raise ImportError("momentfm is required for MOMENT embeddings. Install with: pip install momentfm") from None

    ids, arrays = _extract_series(df, target_col, id_col, time_col)

    model = MOMENTPipeline.from_pretrained(model_name, model_task="embedding")
    model.init()
    model = model.to(device)

    # Pad/truncate to uniform length and stack
    max_len = max(a.shape[0] for a in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, a in enumerate(arrays):
        padded[i, : a.shape[0]] = a

    with torch.no_grad():
        input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, T)
        output = model(input_tensor)
        embeddings = output.embeddings.cpu().numpy()  # (N, d)

    return _arrays_to_result(ids, embeddings, id_col, "emb")
