## Unreleased

### Improvements

- CI: replace `pre-commit` with [prek](https://github.com/j178/prek) (Rust reimplementation) for faster code-quality checks.
- CI: add [ty](https://github.com/astral-sh/ty) type checker as non-blocking informational job alongside mypy.
- Add `[tool.ty]` configuration section in `pyproject.toml`.
- Document `prek` and `ty` local developer workflows in README.

## v0.5.0 (2026-04-16)

### Features

- Add KShape time series clustering using shape-based distance with centroid computation.
- Add KShape time series classifier.
- Add k-Medoids (PAM) time series clustering (`kmedoids`) using any of the 12 distance metrics.
- Add k-Nearest Neighbors time series classification (`knn_classify`) using any of the 12 distance metrics.
- Add SBD (Shape-Based Distance) metric (`compute_pairwise_sbd`).
- Add Frechet distance metric (`compute_pairwise_frechet`).
- Add EDR (Edit Distance on Real Sequences) metric (`compute_pairwise_edr`).

### Improvements

- Add shared distance dispatch utility (`_distance_dispatch`) for reuse across clustering and classification.
- Upgrade Rust dependencies: pyo3 0.25, polars crate 0.49.1.
- Add `py.typed` marker for PEP 561 type hint distribution.
- Add lazy import system for optional dependencies (`forecast`, `decomposition`).
- CI: add coverage reporting, MkDocs deployment workflow, polars compatibility matrix (1.30–1.33).

### Tests

- Add 85 tests for k-NN classification covering correctness, custom columns, and multiple metrics.
- Add 84 tests for k-Medoids clustering covering output, correctness, edge cases, and multiple metrics.
- Add 90 tests for KShape clustering.
- Add 85 tests for KShape classifier.
- Add tests for SBD, Frechet, EDR distance metrics.
- Add unified API tests for all 12 distance metrics.
- Add 52 lazy import tests for optional dependency handling.

## v0.4.0 (2026-04-14)

### Features

- Add Sen's slope estimator (`sens_slope`) — robust median-of-pairwise-slopes trend magnitude.
- Add CUSUM changepoint detection (`cusum`) — cumulative sum control chart for mean shift detection.
- Add anomaly flagging to decomposition methods via `anomaly_threshold` parameter.
- Add ERP (Edit Distance with Real Penalty) distance metric (`compute_pairwise_erp`).
- Add LCSS (Longest Common Subsequence) distance metric (`compute_pairwise_lcss`).
- Add TWE (Time Warp Edit Distance) distance metric (`compute_pairwise_twe`).
- Add FastDTW approximate algorithm (`method="fast"`, `param=radius`).
- Add Sakoe-Chiba band constraint (`method="sakoe_chiba"`, `param=window_size`).
- Add Itakura parallelogram constraint (`method="itakura"`, `param=max_slope`).
- `compute_pairwise_dtw` now accepts optional `method` and `param` arguments (backward compatible).

### Improvements

- Add unified `compute_pairwise_distance(method=...)` API — single entry point for all 9 distance metrics.
- Deduplicate Rust distance code into shared `utils.rs` (grouping, hashing, parallel pairwise).
- Optimize all distance metrics to O(m) memory with two-row DP.
- Add `#[pyo3(signature)]` annotations for Rust safety with pyo3 0.24+.
- Add `pyarrow` as core dependency for reliable Arrow interchange.
- Lightweight core: optional dependencies (`forecast`, `decomposition`) are lazy-loaded.
- CI: add minimal-deps job, polars compatibility matrix, clippy, and code-quality checks.

### Fixes

- Upgrade Rust dependencies (pyo3 0.24, pyo3-polars 0.21, polars crate 0.48).
- Pin Python polars to `>=1.30.0,<2.0.0` for ABI compatibility.
- Fix trailing space in `freqs` docstring causing mkdocs strict build failure.

### Tests

- Add 286 tests covering all distance metrics, decomposition, trend, changepoint, and edge cases.
- Add input validation, edge case, and parallelism stress tests for distance metrics.

## v0.3.0

### Features

- Implement Seasonal Decomposition.
- Implement Fourier Decomposition.
- Implement Naive Dynamic Time Warping.

## v0.2.0

### Features

- Implement Mann-Kendall's Trend Statistic.

### Chore

- Make library usable on PyPI with the Rust expressions.

## v0.1.0

### Features

- Implement Kaboudan metric.

### Documentation

- Add automatic references to docstrings.
- Access docs under [https://drumtorben.github.io/polars-ts/](https://drumtorben.github.io/polars-ts/).

### Chore

- Initialize Repo.
