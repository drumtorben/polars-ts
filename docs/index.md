# polars-ts — Time Series Toolkit for Polars

**polars-ts** is a batteries-included time series toolkit built on [Polars](https://pola.rs/). It gives you Rust-accelerated distance metrics, 10+ clustering algorithms, a full forecasting stack, and diagnostics — all from a single `pip install`.

---

**Documentation**: [https://drumtorben.github.io/polars-ts](https://drumtorben.github.io/polars-ts)

**Source Code**: [https://github.com/drumtorben/polars-ts](https://github.com/drumtorben/polars-ts)

**PyPI**: [https://pypi.org/project/polars-timeseries](https://pypi.org/project/polars-timeseries)

---

## Why polars-ts?

| Pain point | How polars-ts helps |
|---|---|
| "I need DTW but scipy is slow" | 12 distance metrics compiled to native code via Rust + Rayon |
| "I want to cluster time series but tslearn has too many deps" | K-Medoids, K-Shape, HDBSCAN, Spectral, Hierarchical + 6 more — all built-in |
| "Setting up a forecast pipeline takes too long" | `ForecastPipeline` wires up lags, rolling stats, transforms, and any sklearn model in 5 lines |
| "I don't know which clustering method to pick" | `auto_cluster` sweeps methods × distances × k and returns the best |
| "Polars doesn't have time series functions" | Mann-Kendall, Sen's slope, CUSUM, PELT, decomposition, ACF/PACF — all Polars-native |

## Installation

=== "uv (recommended)"
    ```bash
    uv add polars-timeseries
    ```

=== "pip"
    ```bash
    pip install polars-timeseries
    ```

Extras for optional features:

```bash
pip install "polars-timeseries[clustering]"     # HDBSCAN, DBSCAN, spectral
pip install "polars-timeseries[forecast]"       # SCUM, auto_arima
pip install "polars-timeseries[all]"            # Everything
```

Requires **Python 3.12+** and **Polars 1.30+**.

## Quick start

### Cluster time series automatically

```python
import polars_ts as pts

result = pts.auto_cluster(
    df,
    methods=["kmedoids", "spectral", "kshape"],
    distances=["sbd", "dtw"],
    k_range=range(2, 6),
)
print(result.best_method, result.best_k, result.best_score)
```

### Build a forecast pipeline

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
```

### Compute pairwise DTW distances

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

## What's included

| Category | Highlights |
|---|---|
| [**Distance metrics**](distance-metrics.md) | 12 Rust-accelerated metrics (DTW, SBD, MSM, ERP, ...) |
| [**Clustering**](clustering-guide.md) | K-Medoids, K-Shape, HDBSCAN, DBSCAN, Spectral, Hierarchical, K-Means DBA, CLARA, CLARANS, U-Shapelets, auto_cluster |
| [**Forecasting**](forecasting-guide.md) | Baselines, ARIMA, exponential smoothing, ML pipelines, global models, ensembles |
| [**Imaging**](imaging-guide.md) | Recurrence plots, GAF, MTF, spectrograms, scalograms, vision model embeddings |
| [**Changepoint & anomaly**](changepoint-guide.md) | CUSUM, PELT, BOCPD, regime detection, Isolation Forest |
| [**Preprocessing**](preprocessing-guide.md) | Imputation, outlier detection, resampling, feature engineering, target transforms |

## Tutorials

Interactive notebooks covering the full toolkit:

| # | Topic | Notebook |
|---|---|---|
| 01 | Data wrangling & exploration | [01_data_wrangling_and_exploration.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/01_data_wrangling_and_exploration.ipynb) |
| 02 | Feature engineering & transforms | [02_feature_engineering_transforms.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/02_feature_engineering_transforms.ipynb) |
| 03 | Forecasting fundamentals | [03_forecasting_fundamentals.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/03_forecasting_fundamentals.ipynb) |
| 04 | ML forecasting pipelines | [04_ml_forecasting_pipelines.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/04_ml_forecasting_pipelines.ipynb) |
| 05 | Uncertainty & calibration | [05_uncertainty_and_calibration.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/05_uncertainty_and_calibration.ipynb) |
| 06 | Changepoint & anomaly detection | [06_changepoint_anomaly_detection.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/06_changepoint_anomaly_detection.ipynb) |
| 07 | Time series similarity & clustering | [07_time_series_similarity_clustering.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/07_time_series_similarity_clustering.ipynb) |
| 08 | Multivariate & volatility | [08_multivariate_volatility.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/08_multivariate_volatility.ipynb) |
| 09 | Ensembles & reconciliation | [09_ensembles_reconciliation.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/09_ensembles_reconciliation.ipynb) |
| 10 | Ecosystem adapters | [10_ecosystem_adapters.ipynb](https://github.com/drumtorben/polars-ts/blob/main/notebooks/10_ecosystem_adapters.ipynb) |
