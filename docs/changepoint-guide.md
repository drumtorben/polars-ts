# Changepoint & Anomaly Detection Guide

polars-ts provides trend tests, changepoint detection, regime inference, and anomaly detection — all group-aware and Polars-native.

## Trend detection

### Mann-Kendall test

Non-parametric test for monotonic trend. Returns a struct with `tau`, `p_value`, and `trend` (increasing/decreasing/no trend). Implemented in Rust.

```python
import polars as pl
import polars_ts as pts

result = df.group_by("unique_id").agg(
    pts.mann_kendall(pl.col("y")).alias("trend"),
)
```

### Sen's slope

Robust trend magnitude estimator (median of all pairwise slopes). Implemented in Rust.

```python
result = df.group_by("unique_id").agg(
    pts.sens_slope(pl.col("y")).alias("slope"),
)
```

## Changepoint detection

### CUSUM

Cumulative sum control chart for detecting mean shifts. Implemented in Rust.

```python
result = pts.cusum(df)
```

### PELT

Pruned Exact Linear Time algorithm for finding multiple changepoints with configurable cost functions.

```python
result = pts.pelt(df, cost="meanvar", pen=10)
```

Cost functions: `"mean"`, `"var"`, `"meanvar"`.

### BOCPD

Bayesian Online Changepoint Detection — detects changepoints in a streaming fashion using a Student-t model.

```python
result = pts.bocpd(df, hazard=250, mu0=0, kappa0=1, alpha0=1, beta0=1)
```

## Regime detection

Hidden Markov Model state inference via Baum-Welch EM.

```python
result = pts.regime_detect(df, n_regimes=3)
```

## Anomaly detection

### Decomposition-based

Flag anomalies from decomposition residuals exceeding a threshold.

```python
result = pts.seasonal_decomposition(df, freq=24, method="additive", anomaly_threshold=2.0)
```

### Isolation Forest

Unsupervised anomaly detection on engineered features.

```python
result = pts.isolation_forest_detect(df, contamination=0.05)
```

## Further reading

- **Notebook 06**: [Changepoint & anomaly detection](https://github.com/drumtorben/polars-ts/blob/main/notebooks/06_changepoint_anomaly_detection.ipynb)
