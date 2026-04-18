# polars-ts v0.6.0 (2026-04-17)

## Features

### Feature Engineering
- **lag_features** — create lagged versions of a target column per group
- **rolling_features** — rolling window aggregations (mean, std, min, max, sum, median, var)
- **calendar_features** — extract day_of_week, month, quarter, is_weekend, etc. from datetime columns
- **fourier_features** — sin/cos pairs for seasonal modelling with configurable harmonics

### Target Transforms
- **log_transform / inverse_log_transform** — natural log with automatic validation for non-positive values
- **boxcox_transform / inverse_boxcox_transform** — parametric power transform (lambda == 0 → log, lambda != 0 → power)
- **difference / undifference** — differencing with configurable order and seasonal period, stores metadata for lossless inversion

All transforms are group-aware (`id_col`), invertible, and accessible as standalone functions or via the `df.pts` namespace.

### Validation Strategies
- **expanding_window_cv** — growing training window time series cross-validation
- **sliding_window_cv** — fixed-size training window cross-validation
- **rolling_origin_cv** — general rolling-origin CV with configurable initial/fixed train size and gap

### Baseline Forecast Models
- **naive_forecast** — repeat the last observed value for h steps
- **seasonal_naive_forecast** — repeat the last season's values cyclically
- **moving_average_forecast** — flat forecast from the mean of the last window_size observations
- **fft_forecast** — FFT-based forecast using dominant frequency components

### Multi-Step Forecasting Strategies
- **RecursiveForecaster** — trains a single 1-step model; feeds predictions back as input for subsequent steps
- **DirectForecaster** — trains h separate models, one per forecast horizon step

## Improvements

- Improved type annotations throughout (e.g., `Callable[[pl.Expr], pl.Expr]` for calendar extractors, `float | None` for min_val)
- Updated `pl.Float64` → `pl.Float64()` for newer Polars API compatibility
- All new modules follow consistent group-aware temporal patterns

---

# polars-ts v0.5.0 (2026-04-16)

## Features

- **KShape clustering** — shape-based distance time series clustering with centroid computation
- **KShape classifier** — time series classification using KShape
- **k-Medoids (PAM) clustering** (`kmedoids`) — supports all 12 distance metrics
- **k-Nearest Neighbors classification** (`knn_classify`) — supports all 12 distance metrics
- **3 new distance metrics**: SBD (Shape-Based Distance), Frechet distance, EDR (Edit Distance on Real Sequences)

## Improvements

- Shared distance dispatch utility (`_distance_dispatch`) for reuse across clustering and classification
- Upgraded Rust dependencies: pyo3 0.25, polars crate 0.49.1
- Added `py.typed` marker for PEP 561 type hint distribution
- Lazy import system for optional dependencies (`forecast`, `decomposition`)
- CI: coverage reporting, MkDocs deployment workflow, polars compatibility matrix (1.30–1.33)

## Tests

- 396 new tests covering k-NN classification, k-Medoids clustering, KShape clustering/classification, SBD/Frechet/EDR distance metrics, unified distance API, and lazy imports
