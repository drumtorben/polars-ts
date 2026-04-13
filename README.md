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

### Trend & Changepoint Detection (Rust)

- **Mann-Kendall test** &mdash; non-parametric trend detection
- **Sen's slope** &mdash; robust trend magnitude estimation
- **CUSUM** &mdash; cumulative sum changepoint detection

### Decomposition (Python)

- **Seasonal decomposition** &mdash; additive or multiplicative (classical)
- **Fourier decomposition** &mdash; harmonic decomposition with configurable frequencies
- **Decomposition features** &mdash; trend/seasonal strength extraction (simple or MSTL)
- **Anomaly flagging** &mdash; residual-based anomaly detection from any decomposition

### Forecasting

- **SCUM** &mdash; ensemble model combining AutoARIMA, AutoETS, AutoCES, and DynamicOptimizedTheta
- **Kaboudan metric** &mdash; model robustness evaluation via block-shuffle backtesting

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

# Standard DTW
result = pts.compute_pairwise_dtw(df, df)

# With Sakoe-Chiba band constraint
result = pts.compute_pairwise_dtw(df, df, method="sakoe_chiba", param=2)
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

### CUSUM changepoint detection

```python
import polars as pl
import polars_ts as pts

df = pl.DataFrame({
    "unique_id": ["A"] * 20,
    "y": [1.0] * 10 + [5.0] * 10,
})

result = pts.cusum(df)
```

### Kaboudan metric

```python
import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, OptimizedTheta
import polars_ts as pts  # noqa

df = (
    pl.scan_parquet("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet")
    .filter(pl.col("unique_id").is_in(["H1", "H2", "H3"]))
    .collect()
)

sf = StatsForecast(
    models=[OptimizedTheta(season_length=24), AutoETS(season_length=24)],
    freq=1, n_jobs=-1,
)

res = df.pts.kaboudan(sf, block_size=200, backtesting_start=0.5, n_folds=10)
```

## Development

```bash
git clone https://github.com/drumtorben/polars-ts.git
cd polars-ts
uv sync
uv pip install -e .
uv run pytest
```

## License

MIT
