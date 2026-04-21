# Polars Time Series Extension

A high-performance time series analysis toolkit for [Polars](https://pola.rs), with Rust-powered distance metrics and Python analytics.

---

**Documentation**: [https://drumtorben.github.io/polars-ts](https://drumtorben.github.io/polars-ts)

**Source Code**: [https://github.com/drumtorben/polars-ts](https://github.com/drumtorben/polars-ts)

**PyPI**: [https://pypi.org/project/polars-timeseries](https://pypi.org/project/polars-timeseries)

---

## Features

### Distance Metrics (Rust, parallelized via Rayon)

| Metric | Function | Key Parameters |
|--------|----------|----------------|
| Dynamic Time Warping | `compute_pairwise_dtw` | `method`: standard, sakoe_chiba, itakura, fast |
| Derivative DTW | `compute_pairwise_ddtw` | Shape-sensitive comparison |
| Weighted DTW | `compute_pairwise_wdtw` | `g`: weight sharpness |
| Move-Split-Merge | `compute_pairwise_msm` | `c`: move cost |
| Edit Distance (Real Penalty) | `compute_pairwise_erp` | `g`: gap value |
| Longest Common Subsequence | `compute_pairwise_lcss` | `epsilon`: matching threshold |
| Time Warp Edit Distance | `compute_pairwise_twe` | `nu`: stiffness, `lambda_`: deletion cost |
| Multivariate DTW | `compute_pairwise_dtw_multi` | `metric`: manhattan, euclidean |
| Multivariate MSM | `compute_pairwise_msm_multi` | `c`: move cost |

### Trend & Changepoint Detection

- **Mann-Kendall test** &mdash; non-parametric trend detection (Rust)
- **Sen's slope** &mdash; robust trend magnitude estimation (Rust)
- **CUSUM** &mdash; cumulative sum changepoint detection (Rust)
- **PELT** &mdash; multiple changepoints with mean/variance/meanvar cost functions
- **BOCPD** &mdash; Bayesian Online Changepoint Detection
- **Regime detection** &mdash; Hidden Markov Model state inference

### Decomposition (Python)

- **Seasonal decomposition** &mdash; additive or multiplicative (classical)
- **Fourier decomposition** &mdash; harmonic decomposition with configurable frequencies
- **Decomposition features** &mdash; trend/seasonal strength extraction (simple or MSTL)
- **Anomaly flagging** &mdash; residual-based anomaly detection from any decomposition

### Feature Engineering

- **Lag features** &mdash; create lagged versions of a target column per group
- **Rolling features** &mdash; rolling window aggregations (mean, std, min, max, sum, median, var)
- **Calendar features** &mdash; extract day_of_week, month, quarter, is_weekend, etc.
- **Fourier features** &mdash; sin/cos pairs for seasonal modelling
- **Target encoding** &mdash; smoothed categorical encoding by target mean
- **Holiday features** &mdash; binary holidays + distance-to-holiday (requires `holidays` package)
- **Interaction features** &mdash; cross-term column generation
- **Time embeddings** &mdash; cyclical sin/cos encoding for time components

### Target Transforms

- **Log transform** &mdash; log1p / expm1 with automatic validation and lossless inversion
- **Box-Cox transform** &mdash; parametric power transform with configurable lambda
- **Differencing** &mdash; configurable order and seasonal period with metadata for lossless inversion

All transforms are group-aware, invertible, and accessible via the `df.pts` namespace.

### Data Preprocessing

- **Missing value imputation** &mdash; forward/backward fill, linear interpolation, mean, median, seasonal
- **Outlier detection** &mdash; z-score, IQR, Hampel filter, rolling z-score
- **Outlier treatment** &mdash; clip (winsorize), median replacement, interpolation, null
- **Temporal resampling** &mdash; downsample/upsample with configurable aggregation

### Validation Strategies

- **Expanding window CV** &mdash; growing training window cross-validation
- **Sliding window CV** &mdash; fixed-size training window cross-validation
- **Rolling origin CV** &mdash; general rolling-origin with configurable initial/fixed train size

### Forecasting

- **SCUM** &mdash; ensemble model combining AutoARIMA, AutoETS, AutoCES, and DynamicOptimizedTheta
- **Baseline models** &mdash; naive, seasonal naive, moving average, and FFT-based forecasts
- **Exponential smoothing** &mdash; SES, Holt's linear, Holt-Winters (additive/multiplicative)
- **Multi-step strategies** &mdash; `RecursiveForecaster` and `DirectForecaster`
- **ForecastPipeline** &mdash; end-to-end ML pipeline with feature engineering + transforms
- **GlobalForecaster** &mdash; cross-series panel model with optional ID encoding

### Probabilistic Forecasting

- **QuantileRegressor** &mdash; one model per quantile level with CRPS-compatible output
- **Conformal prediction** &mdash; distribution-free intervals with coverage guarantees
- **EnbPI** &mdash; Ensemble Batch Prediction Intervals with adaptive online updates

### Ensembling

- **WeightedEnsemble** &mdash; equal, manual, or inverse-error-optimized weights
- **StackingForecaster** &mdash; meta-learner trained on out-of-fold predictions

### Forecast Evaluation & Diagnostics

- **Metrics** &mdash; MAE, RMSE, MAPE, sMAPE, MASE, CRPS
- **Kaboudan metric** &mdash; model robustness evaluation via block-shuffle backtesting
- **Bias detection & correction** &mdash; mean, regression, quantile mapping
- **Calibration diagnostics** &mdash; calibration table, PIT histogram, reliability diagram
- **Residual diagnostics** &mdash; ACF, PACF, Ljung-Box test
- **Permutation importance** &mdash; model-agnostic feature importance

### Multivariate & Hierarchical

- **VAR** &mdash; Vector Autoregression with OLS fitting and multi-step forecasts
- **Granger causality** &mdash; F-test for causal relationships between series
- **GARCH** &mdash; volatility modelling and conditional variance forecasting
- **Forecast reconciliation** &mdash; bottom-up, top-down, and MinTrace-OLS

### Clustering & Classification

- **k-Medoids (PAM)** &mdash; supports all 12 distance metrics
- **KShape** &mdash; shape-based clustering with centroid computation
- **k-NN classification** &mdash; distance-based time series classification
- **KShape classifier** &mdash; classification using KShape centroids

### Anomaly Detection

- **Decomposition-based** &mdash; residual threshold anomaly flagging
- **Isolation Forest** &mdash; unsupervised anomaly detection on engineered features

### Integration Adapters

- **neuralforecast** &mdash; convert to/from N-BEATS, PatchTST, N-HiTS format
- **pytorch-forecasting** &mdash; convert to/from TFT, DeepAR format
- **HuggingFace** &mdash; convert to Dataset for Chronos, TimesFM, Lag-Llama
- **ForecastEnv** &mdash; gymnasium-compatible RL environment for decision making

## Installation

```bash
pip install polars-timeseries
```

Optional dependencies for specific features:

```bash
pip install "polars-timeseries[forecast]"      # Kaboudan metric, SCUM model
pip install "polars-timeseries[decomposition]"  # Fourier decomposition
pip install "polars-timeseries[all]"            # Everything
```

Requires Python 3.12+ and Polars 1.30+.

## Quick Start

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
```

### Exponential smoothing

```python
import polars_ts as pts

# Holt-Winters seasonal forecast
result = pts.holt_winters_forecast(df, h=12, season_length=12, seasonal="additive")
```

### Conformal prediction intervals

```python
import polars_ts as pts

# Distribution-free prediction intervals
result = pts.conformal_interval(cal_residuals, predictions, coverage=0.9)
```

### Weighted ensemble

```python
import polars_ts as pts

ens = pts.WeightedEnsemble(weights="inverse_error")
combined = ens.combine([forecast_a, forecast_b], validation_dfs=[val_a, val_b])
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
import polars as pl
import polars_ts as pts

df = pl.DataFrame({
    "unique_id": ["A"] * 48,
    "ds": list(range(48)),
    "y": [10 + 5 * (i % 12 > 5) + 0.5 * i for i in range(48)],
})

result = pts.seasonal_decomposition(df, freq=12, method="additive")
```

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
