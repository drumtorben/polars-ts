"""ROCKET and MiniRocket feature extraction for time series.

Transforms each time series into a fixed-length feature vector using random
convolutional kernels, enabling fast and accurate clustering/classification
in the resulting feature space.

References
----------
- Dempster et al. (2020) ROCKET: Exceptionally fast and accurate time series
  classification using random convolutional kernels. DMKD.
- Dempster et al. (2021) MiniRocket: A very fast (almost) deterministic
  transform for time series classification. KDD.

"""

from __future__ import annotations

import numpy as np
import polars as pl


def _extract_series(
    df: pl.DataFrame,
    target_col: str,
    id_col: str,
    time_col: str,
) -> tuple[list[str], np.ndarray]:
    """Extract series as a 2-D array (n_series, max_len), zero-padded."""
    sorted_df = df.sort(id_col, time_col)
    groups = sorted_df.group_by(id_col, maintain_order=True)
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    for key, group in groups:
        ids.append(key[0] if isinstance(key, tuple) else key)  # type: ignore[arg-type]
        arrays.append(group[target_col].to_numpy().astype(np.float64))

    max_len = max(a.shape[0] for a in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.float64)
    for i, a in enumerate(arrays):
        padded[i, : a.shape[0]] = a
    return ids, padded


def _generate_rocket_kernels(
    n_kernels: int,
    input_length: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, float, int]]:
    """Generate random convolutional kernels for ROCKET.

    Returns list of (weights, bias, dilation) tuples.
    """
    candidate_lengths = np.array([7, 9, 11])
    kernels: list[tuple[np.ndarray, float, int]] = []
    for _ in range(n_kernels):
        length = rng.choice(candidate_lengths)
        weights = rng.standard_normal(length)
        weights = weights - weights.mean()
        max_dilation = max(1, (input_length - 1) // (length - 1))
        dilation = int(2 ** rng.uniform(0, np.log2(max_dilation + 1)))
        dilation = min(dilation, max_dilation)
        bias = rng.uniform(-1, 1)
        kernels.append((weights, bias, dilation))
    return kernels


def _apply_kernel(x: np.ndarray, weights: np.ndarray, bias: float, dilation: int) -> tuple[float, float]:
    """Apply a single kernel to a 1-D series, return (ppv, max_val)."""
    conv = _convolve_1d(x, weights, dilation)
    if len(conv) == 0:
        return 0.0, 0.0
    conv += bias
    ppv = float(np.mean(conv > 0))
    max_val = float(np.max(conv))
    return ppv, max_val


def rocket_features(
    df: pl.DataFrame,
    n_kernels: int = 500,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    seed: int = 42,
) -> pl.DataFrame:
    """Extract ROCKET features from time series.

    Generates *n_kernels* random convolutional kernels with varying length,
    dilation, and bias, then computes PPV (proportion of positive values) and
    global max for each kernel on every series.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    n_kernels
        Number of random convolutional kernels.
    target_col
        Column with the values to transform.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    seed
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, rocket_0, rocket_1, ...,
        rocket_{2*n_kernels - 1}]``.

    """
    if n_kernels < 1:
        raise ValueError("n_kernels must be at least 1")

    ids, X = _extract_series(df, target_col, id_col, time_col)
    rng = np.random.default_rng(seed)
    kernels = _generate_rocket_kernels(n_kernels, X.shape[1], rng)

    n_series = X.shape[0]
    features = np.empty((n_series, 2 * n_kernels), dtype=np.float64)
    for k_idx, (weights, bias, dilation) in enumerate(kernels):
        for s_idx in range(n_series):
            ppv, max_val = _apply_kernel(X[s_idx], weights, bias, dilation)
            features[s_idx, 2 * k_idx] = ppv
            features[s_idx, 2 * k_idx + 1] = max_val

    col_names = [f"rocket_{i}" for i in range(2 * n_kernels)]
    result = pl.DataFrame({id_col: ids})
    feat_df = pl.DataFrame({name: features[:, i] for i, name in enumerate(col_names)})
    return pl.concat([result, feat_df], how="horizontal")


# ── MiniRocket ──────────────────────────────────────────────────────────


_MINIROCKET_KERNEL_LENGTH = 9


def _get_minirocket_patterns() -> list[np.ndarray]:
    """Return the 84 fixed MiniRocket weight patterns.

    Uses the 84 unique patterns from C(9,3) = 84 combinations. Each
    pattern has weights of 2 at selected positions and -1 elsewhere,
    then mean-centered.
    """
    from itertools import combinations

    patterns: list[np.ndarray] = []
    for combo in combinations(range(9), 3):
        w = np.full(9, -1.0)
        for idx in combo:
            w[idx] = 2.0
        w = w - w.mean()
        patterns.append(w)
    return patterns


def _compute_dilations(input_length: int, n_dilations: int) -> np.ndarray:
    """Compute dilations ensuring the kernel spans the full input length."""
    max_dilation = max(1, (input_length - 1) // (_MINIROCKET_KERNEL_LENGTH - 1))
    dilations = np.unique(np.logspace(0, np.log2(max_dilation), n_dilations, base=2).astype(int))
    return dilations


def minirocket_features(
    df: pl.DataFrame,
    n_kernels: int = 500,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
    seed: int = 42,
) -> pl.DataFrame:
    """Extract MiniRocket features from time series.

    Uses fixed-length (9) kernels with deterministic weight patterns and
    learned biases.  Significantly faster than ROCKET while achieving
    comparable accuracy.

    Parameters
    ----------
    df
        Input DataFrame with time series data.
    n_kernels
        Approximate number of kernels (rounded to nearest multiple of 84).
    target_col
        Column with the values to transform.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    seed
        Random seed for bias sampling.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, minirocket_0, ...,
        minirocket_{n_features - 1}]``.

    """
    if n_kernels < 1:
        raise ValueError("n_kernels must be at least 1")

    ids, X = _extract_series(df, target_col, id_col, time_col)
    n_series, input_length = X.shape
    rng = np.random.default_rng(seed)

    patterns = _get_minirocket_patterns()  # 84 patterns
    n_patterns = len(patterns)
    n_dilations = max(1, n_kernels // n_patterns)
    dilations = _compute_dilations(input_length, n_dilations)

    # Collect all (pattern, dilation, bias) combinations
    kernel_configs: list[tuple[np.ndarray, int, float]] = []
    for dilation in dilations:
        for pattern in patterns:
            # Compute convolution outputs for all series to sample bias
            sample_idx = rng.integers(0, n_series)
            conv_out = _convolve_1d(X[sample_idx], pattern, int(dilation))
            if len(conv_out) > 0:
                bias = float(rng.choice(conv_out))
            else:
                bias = 0.0
            kernel_configs.append((pattern, int(dilation), bias))

    # Trim to requested number of kernels
    if len(kernel_configs) > n_kernels:
        kernel_configs = kernel_configs[:n_kernels]

    n_features = len(kernel_configs)
    features = np.empty((n_series, n_features), dtype=np.float64)
    for k_idx, (weights, dilation, bias) in enumerate(kernel_configs):
        for s_idx in range(n_series):
            conv = _convolve_1d(X[s_idx], weights, dilation)
            if len(conv) > 0:
                features[s_idx, k_idx] = float(np.mean((conv > bias)))
            else:
                features[s_idx, k_idx] = 0.0

    col_names = [f"minirocket_{i}" for i in range(n_features)]
    result = pl.DataFrame({id_col: ids})
    feat_df = pl.DataFrame({name: features[:, i] for i, name in enumerate(col_names)})
    return pl.concat([result, feat_df], how="horizontal")


def _convolve_1d(x: np.ndarray, weights: np.ndarray, dilation: int) -> np.ndarray:
    """1-D dilated convolution (no bias) returning the raw output."""
    k_len = len(weights)
    output_len = len(x) - (k_len - 1) * dilation
    if output_len <= 0:
        return np.empty(0)
    # Gather dilated indices: shape (output_len, k_len)
    offsets = np.arange(k_len) * dilation  # (k_len,)
    starts = np.arange(output_len)[:, None]  # (output_len, 1)
    indices = starts + offsets  # (output_len, k_len)
    return x[indices] @ weights  # (output_len,)
