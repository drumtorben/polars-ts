## v0.3.2 (Unreleased)

### Features

- Add FastDTW approximate algorithm (`method="fast"`, `param=radius`).
- Add Sakoe-Chiba band constraint (`method="sakoe_chiba"`, `param=window_size`).
- Add Itakura parallelogram constraint (`method="itakura"`, `param=max_slope`).
- `compute_pairwise_dtw` now accepts optional `method` and `param` arguments (backward compatible).

### Fixes

- Upgrade Rust dependencies for Python polars compatibility (pyo3 0.23 to 0.24, pyo3-polars 0.20 to 0.21, polars crate 0.46 to 0.48).
- Add `#[pyo3(signature)]` annotations to fix implicit `Option` defaults in pyo3 0.24+ (wdtw, msm, msm_multi, dtw_multi).
- Remove unused `polars-rows-iter` dependency.
- Fix trailing space in `freqs` docstring parameter causing mkdocs strict build failure.
- Pin Python polars to `>=1.20.0,<1.32.3` for ABI compatibility with Rust polars 0.48.

## v0.3.0

### ✨ Features

- Implement Seasonal Decomposition.
- Implement Fourier Decomposition.
- Implement Naive Dynamic Time Warping.

## v0.2.0

### ✨ Features

- Implement Mann-Kendall's Trend Statistic.

### 🛠️ Chore

- Make library usable on PyPI with the Rust expressions.

## v0.1.0

### ✨ Features

- Implement Kaboudan metric.

### 📖 Documentation

- Add automatic references to docstrings.
- Access docs under [https://drumtorben.github.io/polars-ts/](https://drumtorben.github.io/polars-ts/).

### 🛠️ Chore

- Initialize Repo.
