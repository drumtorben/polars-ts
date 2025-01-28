# Polars Time Series Extension

Welcome to the documentation for the **Polars Time Series Extension**.

---

**Documentation**: [https://drumtorben.github.io/polars-ts](https://drumtorben.github.io/polars-ts)

**Source Code**: [https://github.com/drumtorben/polars-ts](https://github.com/drumtorben/polars-ts)

---

## 📖 Overview

The **Polars Time Series Extention** offers a wide range of metrics, feature extractors, and various tools for time series forecasting.

## Installation

`pip install polars-timeseries`

## How to use

The `polars-ts` plugin is available under the namespace `pts`.
See the following example where we compute the Kaboudan metric:

```python
import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, OptimizedTheta

import polars_ts as pts  # noqa

# Create sample dataframe with columns `unique_id`, `ds`, and `y`.
df = (
    pl.scan_parquet("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet")
    .filter(pl.col("unique_id").is_in(["H1", "H2", "H3"]))
    .collect()
)

# Define models
season_length = 24
models = [
    OptimizedTheta(season_length=season_length, decomposition_type="additive"),
    AutoETS(season_length=season_length),
]
sf = StatsForecast(models=models, freq=1, n_jobs=-1)

# Compute the Kaboudan metric in the `pts` namespace
res = df.pts.kaboudan(sf, block_size=200, backtesting_start=0.5, n_folds=10)
```