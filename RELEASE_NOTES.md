# polars-ts v0.8.0 (2026-04-25)

The clustering expansion release: polars-ts goes from 2 clustering algorithms to **11**, plus an automated pipeline selector.

## Highlights

- **9 new clustering methods** — HDBSCAN, DBSCAN, spectral (KSC), agglomerative, K-Means DBA, CLARA, CLARANS, U-Shapelets, and `auto_cluster` for automated method/distance/k selection
- **Feature extraction** — ROCKET/MiniRocket random convolutional kernels, Chronos/MOMENT foundation model embeddings
- **2 security patches** — rand 0.8.6 (GHSA-cq8v-f236-94qc), rustls-webpki 0.103.13 (GHSA-82j2-j2ch-gfr8)

## New clustering methods

| Method | Function | Key capability |
|---|---|---|
| **HDBSCAN** | `hdbscan_cluster` | Automatic k, varying density, noise detection |
| **DBSCAN** | `dbscan_cluster` | Fixed-radius eps neighbourhood, noise detection |
| **Spectral (KSC)** | `spectral_cluster` | Graph Laplacian + Gaussian kernel affinity |
| **Hierarchical** | `agglomerative_cluster` | Dendrograms, flexible linkage criteria |
| **K-Means DBA** | `kmeans_dba` | DTW Barycentric Averaging centroids |
| **CLARA** | `clara` | Scalable k-medoids via sampling |
| **CLARANS** | `clarans` | Randomized k-medoids neighbourhood search |
| **U-Shapelets** | `shapelet_cluster` | Interpretable sub-sequence patterns |
| **Auto-cluster** | `auto_cluster` | Sweep methods × distances × k, pick the best |

## Feature extraction for clustering

- **ROCKET / MiniRocket** — random convolutional kernel features (`rocket_features`, `minirocket_features`)
- **Chronos embeddings** — `to_chronos_embeddings` for Amazon Chronos foundation model
- **MOMENT embeddings** — `to_moment_embeddings` for CMU MOMENT foundation model

## Fixes

- Add `scipy>=1.11.0` to `clustering` optional dependency (required by spectral clustering)
- Bump `rand` 0.8.5 → 0.8.6 to resolve GHSA-cq8v-f236-94qc
- Upgrade `rustls-webpki` 0.103.12 → 0.103.13 for GHSA-82j2-j2ch-gfr8

## Documentation

- Replace 7 API-demo notebooks with 10 theme-based tutorials
- Add polars-ts logo for README banner

---

# polars-ts v0.7.0 (2026-04-21)

## CI & Tooling

- **prek** — replaced `pre-commit` with [prek](https://github.com/j178/prek) in CI. 3-5x faster code-quality checks, same `.pre-commit-config.yaml`. Contributors can use either tool locally.
- **ty** — added [ty](https://github.com/astral-sh/ty) (Astral's Rust-based type checker) as a non-blocking CI job (`continue-on-error: true`). Runs alongside mypy as informational. Will be promoted to blocking when ty reaches stable.

## Performance

- Accelerate exponential smoothing (SES, Holt, Holt-Winters) with Rust implementation
- Accelerate k-medoids PAM clustering with Rust implementation

## Features

### Probabilistic Forecasting
- **QuantileRegressor** — trains one model per quantile level; recursive multi-step with CRPS-compatible `q_*` output
- **conformal_interval** — distribution-free prediction intervals with finite-sample correction (symmetric/asymmetric)
- **EnbPI** — Ensemble Batch Prediction Intervals with bootstrap OOB residuals and online `update()`

### Ensembling
- **WeightedEnsemble** — combine forecasts with equal, manual, or inverse-error-optimized weights
- **StackingForecaster** — meta-learner trained on out-of-fold base model predictions

### ML Forecasting Pipeline
- **ForecastPipeline** — end-to-end fit/predict with configurable feature engineering (lags, rolling, calendar, Fourier) and target transforms (log, Box-Cox, differencing)
- **GlobalForecaster** — cross-series panel model with optional series-identity encoding (ordinal/one-hot) and static exogenous features

### Exponential Smoothing
- **ses_forecast** — Simple Exponential Smoothing (level only, flat forecast)
- **holt_forecast** — Holt's linear trend (level + trend)
- **holt_winters_forecast** — Holt-Winters seasonal (additive and multiplicative)

### Data Preprocessing
- **impute** — missing value imputation (forward_fill, backward_fill, linear, mean, median, seasonal)
- **detect_outliers** — outlier detection (z-score, IQR, Hampel filter, rolling z-score)
- **treat_outliers** — outlier treatment (clip, median, interpolate, null)
- **resample** — group-aware temporal resampling with configurable aggregation

### Residual Diagnostics
- **acf** — autocorrelation function with 95% confidence bands
- **pacf** — partial autocorrelation via Durbin-Levinson recursion
- **ljung_box** — Ljung-Box test for residual autocorrelation

### Advanced Changepoint Detection
- **pelt** — PELT algorithm for multiple changepoints (mean/var/meanvar cost functions)
- **bocpd** — Bayesian Online Changepoint Detection with Student-t model
- **regime_detect** — Gaussian HMM regime detection via Baum-Welch EM

### Multivariate & Hierarchical
- **var_fit / var_forecast** — Vector Autoregression (VAR) with OLS fitting
- **granger_causality** — F-test for Granger causality between series
- **garch_fit / garch_forecast** — GARCH(p,q) volatility modelling via MLE
- **reconcile** — forecast reconciliation (bottom-up, top-down, MinTrace-OLS)

### Advanced Feature Engineering
- **target_encode** — smoothed categorical encoding by target mean
- **holiday_features** — binary holidays + distance-to-holiday (requires `holidays`)
- **interaction_features** — cross-term column generation (multiply/add)
- **time_embeddings** — cyclical sin/cos encoding for time components

### Forecast Evaluation
- **bias_detect / bias_correct** — detect and correct systematic forecast bias
- **calibration_table** — observed vs expected coverage per quantile
- **pit_histogram** — Probability Integral Transform histogram
- **reliability_diagram** — data for calibration plots
- **permutation_importance** — model-agnostic feature importance

### Anomaly Detection
- **isolation_forest_detect** — Isolation Forest adapter on engineered features

### Integration Adapters
- **to/from_neuralforecast** — N-BEATS, PatchTST, N-HiTS format conversion
- **to/from_pytorch_forecasting** — TFT, DeepAR format conversion
- **to_hf_dataset** — HuggingFace Dataset for Chronos, TimesFM
- **ForecastEnv** — gymnasium-compatible RL environment

---

# polars-ts v0.6.0 (2026-04-17)

## Features

### Feature Engineering
- **lag_features** — create lagged versions of a target column per group
- **rolling_features** — rolling window aggregations (mean, std, min, max, sum, median, var)
- **calendar_features** — extract day_of_week, month, quarter, is_weekend, etc. from datetime columns
- **fourier_features** — sin/cos pairs for seasonal modelling with configurable harmonics

### Target Transforms
- **log_transform / inverse_log_transform** — natural log with automatic validation for non-positive values
- **boxcox_transform / inverse_boxcox_transform** — parametric power transform
- **difference / undifference** — differencing with configurable order and seasonal period

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
- **RecursiveForecaster** — trains a single 1-step model; feeds predictions back
- **DirectForecaster** — trains h separate models, one per forecast horizon step

---

# polars-ts v0.5.0 (2026-04-16)

## Features

- **KShape clustering** — shape-based distance time series clustering with centroid computation
- **KShape classifier** — time series classification using KShape
- **k-Medoids (PAM) clustering** (`kmedoids`) — supports all 12 distance metrics
- **k-Nearest Neighbors classification** (`knn_classify`) — supports all 12 distance metrics
- **3 new distance metrics**: SBD, Frechet, EDR

## Improvements

- Shared distance dispatch utility for reuse across clustering and classification
- Upgraded Rust dependencies: pyo3 0.25, polars crate 0.49.1
- Added `py.typed` marker for PEP 561 type hint distribution
- Lazy import system for optional dependencies
- CI: coverage reporting, MkDocs deployment workflow, polars compatibility matrix (1.30-1.33)
