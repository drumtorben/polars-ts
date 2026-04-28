"""Path signature features and signature image visualization.

Computes truncated path signatures as feature vectors for time series,
and reshapes them into 2D images for vision-based analysis.

Uses ``iisignature`` when available for fast computation; falls back to
a pure-numpy implementation for basic signatures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from polars_ts.imaging._utils import extract_series


def _time_augment(x: np.ndarray) -> np.ndarray:
    """Prepend a normalised time channel: path becomes 2D (t, x)."""
    n = len(x)
    t = np.linspace(0, 1, n)
    return np.column_stack([t, x])


def _leadlag_augment(x: np.ndarray) -> np.ndarray:
    """Lead-lag transform: (x_t, x_{t-1}) for richer path structure."""
    lead = x[1:]
    lag = x[:-1]
    return np.column_stack([lead, lag])


def _basepoint_augment(path: np.ndarray) -> np.ndarray:
    """Prepend the origin as basepoint."""
    origin = np.zeros((1, path.shape[1]))
    return np.vstack([origin, path])


def _apply_augmentations(
    x: np.ndarray,
    augmentations: list[str],
) -> np.ndarray:
    """Build a multi-dimensional path from a 1D series via augmentations."""
    path = x.reshape(-1, 1)

    for aug in augmentations:
        if aug == "time":
            t = np.linspace(0, 1, len(path)).reshape(-1, 1)
            path = np.hstack([t, path])
        elif aug == "leadlag":
            path = _leadlag_augment(path[:, 0]) if path.shape[1] == 1 else path
        elif aug == "basepoint":
            path = _basepoint_augment(path)
        else:
            raise ValueError(f"Unknown augmentation {aug!r}. Supported: time, leadlag, basepoint")

    return path


def _sig_numpy(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute truncated signature using pure numpy (iterated integrals).

    This is a reference implementation; for production use install iisignature.
    """
    increments = np.diff(path, axis=0)  # (n-1, d)
    d = path.shape[1]

    terms: list[float] = []

    # Depth 1: sum of increments per channel
    s1 = increments.sum(axis=0)  # (d,)
    terms.extend(s1.tolist())

    if depth >= 2:
        # Depth 2: iterated integrals S^{ij} = sum_{s<t} dX^i_s * dX^j_t
        n = len(increments)
        cumsum = np.cumsum(increments, axis=0)
        for i in range(d):
            for j in range(d):
                val = 0.0
                for t in range(1, n):
                    val += cumsum[t - 1, i] * increments[t, j]
                terms.append(val)

    if depth >= 3:
        # Depth 3: S^{ijk}
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    val = 0.0
                    cum_i = 0.0
                    cum_ij = 0.0
                    for t in range(n):
                        cum_ij += cum_i * increments[t, j]
                        val += cum_ij * increments[t, k]
                        cum_i += increments[t, i]
                    terms.append(val)

    if depth >= 4:
        raise ValueError("Pure-numpy signature only supports depth <= 3. Install iisignature for higher depths.")

    return np.array(terms, dtype=np.float64)


def _compute_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute the truncated signature, using iisignature if available."""
    try:
        import iisignature

        s = iisignature.sig(path, depth)
        return np.asarray(s, dtype=np.float64)
    except ImportError:
        return _sig_numpy(path, depth)


def signature_features(
    df: pl.DataFrame,
    depth: int = 3,
    augmentations: list[str] | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pl.DataFrame:
    """Extract truncated path signature features for each time series.

    The signature is a sequence of iterated integrals that captures the
    shape of a path up to reparametrisation. Truncating at depth ``d``
    gives a finite-dimensional feature vector.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    depth
        Truncation depth. Controls feature dimensionality:
        depth 1 → d features, depth 2 → d + d², depth 3 → d + d² + d³.
    augmentations
        List of path augmentations to apply before computing the
        signature. Options: ``"time"`` (prepend time channel),
        ``"leadlag"`` (lead-lag transform), ``"basepoint"`` (prepend origin).
        Default: ``["time"]``.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, sig_0, sig_1, ..., sig_d]``.

    """
    if augmentations is None:
        augmentations = ["time"]

    series = extract_series(df, id_col, target_col)
    all_ids: list[str] = []
    all_sigs: list[np.ndarray] = []

    for sid, vals in series.items():
        path = _apply_augmentations(vals, augmentations)
        sig = _compute_signature(path, depth)
        all_ids.append(sid)
        all_sigs.append(sig)

    embeddings = np.stack(all_sigs)
    n_dim = embeddings.shape[1]
    data: dict[str, Any] = {id_col: all_ids}
    for i in range(n_dim):
        data[f"sig_{i}"] = embeddings[:, i].tolist()

    return pl.DataFrame(data)


def to_signature_image(
    df: pl.DataFrame,
    depth: int = 2,
    augmentations: list[str] | None = None,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to signature images.

    Computes the depth-2 signature and reshapes the d² terms into a
    d × d matrix for visualization. Higher-depth terms are discarded.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    depth
        Signature depth (must be >= 2). Only the depth-2 terms are
        used for the image; depth-1 terms are discarded.
    augmentations
        Path augmentations. Default: ``["time"]``.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a d × d matrix of depth-2 signature terms.

    """
    if depth < 2:
        raise ValueError("depth must be >= 2 for signature images")

    if augmentations is None:
        augmentations = ["time"]

    series = extract_series(df, id_col, target_col)
    result: dict[str, np.ndarray] = {}

    for sid, vals in series.items():
        path = _apply_augmentations(vals, augmentations)
        d = path.shape[1]
        sig = _compute_signature(path, max(depth, 2))
        # Depth-2 terms start at index d and have d² elements
        depth2_terms = sig[d : d + d * d]
        result[sid] = depth2_terms.reshape(d, d)

    return result
