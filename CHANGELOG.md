## v0.4.0 (Unreleased)

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
