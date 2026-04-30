<p align="center">
  <img src="images/polars_ts_complete.png" alt="Polars TS" width="600">
</p>

<p align="center">
  <a href="https://drumtorben.github.io/polars-ts">Documentation</a> &bull;
  <a href="https://github.com/drumtorben/polars-ts">Source Code</a> &bull;
  <a href="https://pypi.org/project/polars-timeseries">PyPI</a>
</p>

<p align="center">
  <a href="https://drumtorben.github.io/polars-ts/distance-metrics/">Distance Metrics</a> &bull;
  <a href="https://drumtorben.github.io/polars-ts/clustering-guide/">Clustering</a> &bull;
  <a href="https://drumtorben.github.io/polars-ts/forecasting-guide/">Forecasting</a> &bull;
  <a href="https://drumtorben.github.io/polars-ts/imaging-guide/">Imaging</a> &bull;
  <a href="https://drumtorben.github.io/polars-ts/changepoint-guide/">Changepoints</a> &bull;
  <a href="https://drumtorben.github.io/polars-ts/preprocessing-guide/">Preprocessing</a>
</p>

---

**polars-ts** is a batteries-included time series toolkit built on [Polars](https://pola.rs/). It gives you Rust-accelerated distance metrics, 10+ clustering algorithms, a full forecasting stack, and diagnostics — all from a single `pip install`, no heavyweight frameworks required.

### Why polars-ts?

| Pain point | How polars-ts helps |
|---|---|
| **"I need DTW but scipy is slow"** | 12 distance metrics compiled to native code via Rust + Rayon, orders of magnitude faster on large panels |
| **"I want to cluster time series but tslearn/sktime have too many deps"** | K-Medoids, K-Shape, HDBSCAN, Spectral, Hierarchical, K-Means DBA, CLARA/CLARANS, U-Shapelets — all built-in, optional `scikit-learn` only for density methods |
| **"Setting up a forecast pipeline takes too long"** | `ForecastPipeline` wires up lags, rolling stats, calendar features, target transforms, and any sklearn model in 5 lines |
| **"I don't know which clustering method to pick"** | `auto_cluster` sweeps methods × distances × k values and returns the best result with evaluation scores |
| **"Polars doesn't have time series functions"** | Mann-Kendall, Sen's slope, CUSUM, PELT, decomposition, ACF/PACF — all group-aware and Polars-native |

### At a glance

```python
import numpy as np
import polars as pl
import polars_ts as pts
from datetime import datetime, timedelta

# Build a small panel of 20 hourly series
dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(100)]
df = pl.concat([
    pl.DataFrame({
        "unique_id": [f"s_{i}"] * 100,
        "ds": dates,
        "y": (np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)).tolist(),
    })
    for i in range(20)
])

# Cluster by shape similarity
result = pts.auto_cluster(df, methods=["kmedoids", "spectral"], distances=["sbd", "dtw"])
print(result.best_labels)  # DataFrame[unique_id, cluster]

# Forecast with a full ML pipeline
from sklearn.linear_model import Ridge
pipe = pts.ForecastPipeline(Ridge(), lags=[1, 7, 14], rolling_windows=[7], calendar=["day_of_week"])
pipe.fit(df)
forecasts = pipe.predict(df, h=7)

# Detect changepoints
breaks = pts.pelt(df, cost="meanvar", penalty=10)
```

> Column defaults are `unique_id`, `ds`, `y` throughout.
> Pass `id_col=`, `time_col=`, `target_col=` to override.

---

## Installation

```bash
pip install polars-timeseries
```

Extras for optional features:

```bash
pip install "polars-timeseries[clustering]"     # HDBSCAN, DBSCAN, spectral (sklearn + scipy)
pip install "polars-timeseries[forecast]"       # SCUM, auto_arima (statsforecast)
pip install "polars-timeseries[decomposition]"  # Fourier decomposition (polars-ds)
pip install "polars-timeseries[all]"            # Everything
```

Requires **Python 3.12+** and **Polars 1.30+**.

---

## Quick start

### Pairwise DTW distance

```python
import polars as pl
import polars_ts as pts

df = pl.DataFrame({
    "unique_id": ["A"] * 5 + ["B"] * 5,
    "y": [1.0, 2.0, 3.0, 2.0, 1.0,
          1.0, 3.0, 5.0, 3.0, 1.0],
})

result = pts.compute_pairwise_dtw(df, df)
```

### Auto-cluster time series

```python
result = pts.auto_cluster(
    df,
    methods=["kmedoids", "spectral", "kshape"],
    distances=["sbd", "dtw"],
    k_range=range(2, 6),
)
print(result.best_method, result.best_k, result.best_score)
print(result.best_labels)  # DataFrame[unique_id, cluster]
```

### End-to-end forecast pipeline

```python
from sklearn.ensemble import GradientBoostingRegressor
import polars_ts as pts

pipe = pts.ForecastPipeline(
    GradientBoostingRegressor(),
    lags=[1, 2, 7],
    rolling_windows=[7],
    calendar=["day_of_week", "month"],
    target_transform="log",
)
pipe.fit(train_df)
forecasts = pipe.predict(train_df, h=7)
# forecasts: DataFrame[unique_id, ds, y_hat]
```

### Forecast with covariates

```python
pipe = pts.ForecastPipeline(
    Ridge(),
    lags=[1, 2, 7],
    past_covariates=["temperature"],       # lagged automatically
    future_covariates=["is_holiday"],       # looked up from future_df
)
pipe.fit(train_df)  # train_df has temperature + is_holiday columns
forecasts = pipe.predict(train_df, h=7, future_df=future_df)
```

### ARIMA forecasting

```python
# Fit ARIMA(1,1,1) and forecast 12 steps ahead
fitted = pts.arima_fit(df, order=(1, 1, 1))
forecast = pts.arima_forecast(fitted, h=12)

# Or use automatic order selection
forecast = pts.auto_arima(df, h=12, season_length=12)
```

### Changepoint detection

```python
# PELT — multiple changepoints with mean/variance cost
breaks = pts.pelt(df, cost="meanvar", penalty=10)
# breaks: DataFrame[unique_id, changepoint_idx, ds]

# Bayesian Online Changepoint Detection
probs = pts.bocpd(df)
```

### Exponential smoothing

```python
# Holt-Winters seasonal forecast
result = pts.holt_winters_forecast(df, h=12, season_length=12, seasonal="additive")
```

### Mann-Kendall trend test

```python
import polars as pl
import polars_ts as pts

df = pl.DataFrame({
    "group": ["A"] * 10 + ["B"] * 10,
    "y": list(range(10)) + [10 - x for x in range(10)],
})

result = df.group_by("group").agg(
    pts.mann_kendall(pl.col("y")).alias("trend"),
    pts.sens_slope(pl.col("y")).alias("slope"),
)
```

### Seasonal decomposition

```python
df = pl.DataFrame({
    "unique_id": ["A"] * 48,
    "ds": list(range(48)),
    "y": [10 + 5 * (i % 12 > 5) + 0.5 * i for i in range(48)],
})

result = pts.seasonal_decomposition(df, freq=12, method="additive")
```

---

## Features

### Distance metrics <sub>Rust, parallelized via Rayon</sub>

All distance functions return a tidy DataFrame with columns `[id_1, id_2, <metric>]`. A unified `compute_pairwise_distance(method=...)` API lets you swap metrics with a single string.

| Metric | Function | Key Parameters |
|--------|----------|----------------|
| Dynamic Time Warping | `compute_pairwise_dtw` | `method`: standard, sakoe_chiba, itakura, fast |
| Derivative DTW | `compute_pairwise_ddtw` | Shape-sensitive comparison |
| Weighted DTW | `compute_pairwise_wdtw` | `g`: weight sharpness |
| Move-Split-Merge | `compute_pairwise_msm` | `c`: move cost |
| Edit Distance (Real Penalty) | `compute_pairwise_erp` | `g`: gap value |
| Longest Common Subsequence | `compute_pairwise_lcss` | `epsilon`: matching threshold |
| Time Warp Edit Distance | `compute_pairwise_twe` | `nu`: stiffness, `lambda_`: deletion cost |
| Shape-Based Distance | `compute_pairwise_sbd` | Cross-correlation based |
| Frechet Distance | `compute_pairwise_frechet` | Geometric coupling distance |
| Edit Distance on Real Sequences | `compute_pairwise_edr` | Edit-operation cost |
| Multivariate DTW | `compute_pairwise_dtw_multi` | `metric`: manhattan, euclidean |
| Multivariate MSM | `compute_pairwise_msm_multi` | `c`: move cost |

### Clustering & classification

| Method | Function | When to use |
|--------|----------|-------------|
| **K-Medoids (PAM)** | `kmedoids` | Known k, any distance metric, interpretable medoids |
| **K-Shape** | `KShape` | Shape-based grouping via cross-correlation centroids |
| **Spectral (KSC)** | `spectral_cluster` | Non-convex clusters, graph Laplacian structure |
| **HDBSCAN** | `hdbscan_cluster` | Unknown k, varying density, noise detection |
| **DBSCAN** | `dbscan_cluster` | Fixed-radius neighbourhood, noise detection |
| **Hierarchical** | `agglomerative_cluster` | Dendrogram visualization, flexible linkage |
| **K-Means DBA** | `kmeans_dba` | DTW Barycentric Averaging centroids |
| **CLARA** | `clara` | Scalable k-medoids via sampling |
| **CLARANS** | `clarans` | Randomized k-medoids neighbourhood search |
| **U-Shapelets** | `shapelet_cluster` | Interpretable sub-sequence patterns |
| **ROCKET / MiniRocket** | `rocket_features`, `minirocket_features` | Random convolutional kernel feature extraction |
| **Auto-cluster** | `auto_cluster` | Sweep methods × distances × k, pick the best |

**Evaluation:** `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`

**Classification:** `knn_classify` (distance-based k-NN), `TimeSeriesKNNClassifier` (OOP), `KShapeClassifier` (centroid-based)

### Trend & changepoint detection

- **Mann-Kendall test** &mdash; non-parametric trend detection (Rust)
- **Sen's slope** &mdash; robust trend magnitude estimation (Rust)
- **CUSUM** &mdash; cumulative sum changepoint detection (Rust)
- **PELT** &mdash; multiple changepoints with mean/variance/meanvar cost functions
- **BOCPD** &mdash; Bayesian Online Changepoint Detection
- **Regime detection** &mdash; Hidden Markov Model state inference

### Decomposition

- **Seasonal decomposition** &mdash; additive or multiplicative (classical)
- **Fourier decomposition** &mdash; harmonic decomposition with configurable frequencies
- **Decomposition features** &mdash; trend/seasonal strength extraction (simple or MSTL)
- **Anomaly flagging** &mdash; residual-based anomaly detection from any decomposition

### Feature engineering

- **Lag features** &mdash; create lagged versions of a target column per group
- **Rolling features** &mdash; rolling window aggregations (mean, std, min, max, sum, median, var)
- **Calendar features** &mdash; extract day_of_week, month, quarter, is_weekend, etc.
- **Fourier features** &mdash; sin/cos pairs for seasonal modelling
- **Target encoding** &mdash; smoothed categorical encoding by target mean
- **Holiday features** &mdash; binary holidays + distance-to-holiday (requires `holidays` package)
- **Interaction features** &mdash; cross-term column generation
- **Time embeddings** &mdash; cyclical sin/cos encoding for time components

### Target transforms

- **Log transform** &mdash; log1p / expm1 with automatic validation and lossless inversion
- **Box-Cox transform** &mdash; parametric power transform with configurable lambda
- **Differencing** &mdash; configurable order and seasonal period with metadata for lossless inversion

All transforms are group-aware, invertible, and accessible via the `df.pts` namespace.

### Data preprocessing

- **Missing value imputation** &mdash; forward/backward fill, linear interpolation, mean, median, seasonal
- **Outlier detection** &mdash; z-score, IQR, Hampel filter, rolling z-score
- **Outlier treatment** &mdash; clip (winsorize), median replacement, interpolation, null
- **Temporal resampling** &mdash; downsample/upsample with configurable aggregation

### Validation strategies

- **Expanding window CV** &mdash; growing training window cross-validation
- **Sliding window CV** &mdash; fixed-size training window cross-validation
- **Rolling origin CV** &mdash; general rolling-origin with configurable initial/fixed train size

### Forecasting

- **SCUM** &mdash; ensemble model combining AutoARIMA, AutoETS, AutoCES, and DynamicOptimizedTheta
- **ARIMA/SARIMA** &mdash; explicit `(p,d,q)` order via `statsmodels` (`arima_fit`/`arima_forecast`) or automatic selection via `statsforecast` (`auto_arima`)
- **Baseline models** &mdash; naive, seasonal naive, moving average, and FFT-based forecasts
- **Exponential smoothing** &mdash; SES, Holt's linear, Holt-Winters (additive/multiplicative, Rust-accelerated)
- **Multi-step strategies** &mdash; `RecursiveForecaster` and `DirectForecaster`
- **ForecastPipeline** &mdash; end-to-end ML pipeline with feature engineering + transforms
- **GlobalForecaster** &mdash; cross-series panel model with optional ID encoding

### Probabilistic forecasting

- **QuantileRegressor** &mdash; one model per quantile level with CRPS-compatible output
- **Conformal prediction** &mdash; distribution-free intervals with coverage guarantees
- **EnbPI** &mdash; Ensemble Batch Prediction Intervals with adaptive online updates

### Ensembling

- **WeightedEnsemble** &mdash; equal, manual, or inverse-error-optimized weights
- **StackingForecaster** &mdash; meta-learner trained on out-of-fold predictions

### Forecast evaluation & diagnostics

- **Metrics** &mdash; MAE, RMSE, MAPE, sMAPE, MASE, CRPS
- **Kaboudan metric** &mdash; model robustness evaluation via block-shuffle backtesting
- **Bias detection & correction** &mdash; mean, regression, quantile mapping
- **Calibration diagnostics** &mdash; calibration table, PIT histogram, reliability diagram
- **Residual diagnostics** &mdash; ACF, PACF, Ljung-Box test
- **Permutation importance** &mdash; model-agnostic feature importance

### Multivariate & hierarchical

- **VAR** &mdash; Vector Autoregression with OLS fitting and multi-step forecasts
- **Granger causality** &mdash; F-test for causal relationships between series
- **GARCH** &mdash; volatility modelling and conditional variance forecasting
- **Forecast reconciliation** &mdash; bottom-up, top-down, and MinTrace-OLS

### Anomaly detection

- **Decomposition-based** &mdash; residual threshold anomaly flagging
- **Isolation Forest** &mdash; unsupervised anomaly detection on engineered features

### Integration adapters

- **NeuralForecast** &mdash; convert to/from N-BEATS, PatchTST, N-HiTS format
- **PyTorch Forecasting** &mdash; convert to/from TFT, DeepAR format
- **HuggingFace** &mdash; convert to Dataset for Chronos, TimesFM, Lag-Llama
- **Chronos / MOMENT embeddings** &mdash; foundation model feature extraction for clustering
- **ForecastEnv** &mdash; Gymnasium-compatible RL environment for decision making

---

## Tutorials

The `notebooks/` directory contains 10 end-to-end tutorials:

| # | Topic | Notebook |
|---|---|---|
| 01 | Data wrangling & exploration | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/01_data_wrangling_and_exploration.ipynb) |
| 02 | Feature engineering & transforms | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/02_feature_engineering_transforms.ipynb) |
| 03 | Forecasting fundamentals | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/03_forecasting_fundamentals.ipynb) |
| 04 | ML forecasting pipelines | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/04_ml_forecasting_pipelines.ipynb) |
| 05 | Uncertainty & calibration | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/05_uncertainty_and_calibration.ipynb) |
| 06 | Changepoint & anomaly detection | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/06_changepoint_anomaly_detection.ipynb) |
| 07 | Time series similarity & clustering | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/07_time_series_similarity_clustering.ipynb) |
| 08 | Multivariate & volatility | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/08_multivariate_volatility.ipynb) |
| 09 | Ensembles & reconciliation | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/09_ensembles_reconciliation.ipynb) |
| 10 | Ecosystem adapters | [Open](https://github.com/drumtorben/polars-ts/blob/main/notebooks/10_ecosystem_adapters.ipynb) |

---

## Development

```bash
git clone https://github.com/drumtorben/polars-ts.git
cd polars-ts
uv sync
uv pip install -e .
uv run pytest
```

### Code quality

Pre-commit hooks run via [prek](https://github.com/j178/prek) (Rust reimplementation of pre-commit) or standard `pre-commit` — both read `.pre-commit-config.yaml`:

```bash
# Option A: prek (faster)
uv tool install prek
prek run --all-files

# Option B: standard pre-commit
pre-commit run --all-files
```

### Type checking

```bash
# mypy (authoritative)
uv run mypy polars_ts/

# ty (fast, informational — beta)
uvx ty check polars_ts/
```

## License

MIT
