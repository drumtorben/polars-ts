# Preprocessing & Feature Engineering Guide

polars-ts includes data preprocessing, feature engineering, target transforms, and validation strategies — all group-aware and designed for time series panels.

## Missing value imputation

```python
import polars_ts as pts

result = pts.impute(df, method="linear")           # linear interpolation
result = pts.impute(df, method="forward_fill")     # forward fill
result = pts.impute(df, method="seasonal", period=24)  # seasonal pattern
```

Methods: `forward_fill`, `backward_fill`, `linear`, `mean`, `median`, `seasonal`.

## Outlier detection & treatment

```python
# Detect
flags = pts.detect_outliers(df, method="zscore", threshold=3.0)
flags = pts.detect_outliers(df, method="iqr", factor=1.5)
flags = pts.detect_outliers(df, method="hampel", window=7)

# Treat
cleaned = pts.treat_outliers(df, method="clip")        # winsorize
cleaned = pts.treat_outliers(df, method="median")      # replace with median
cleaned = pts.treat_outliers(df, method="interpolate")  # linear interpolation
```

## Temporal resampling

```python
result = pts.resample(df, rule="1h", agg="mean")   # downsample to hourly
```

## Feature engineering

```python
# Lag features
result = pts.lag_features(df, lags=[1, 7, 14])

# Rolling features
result = pts.rolling_features(df, windows=[7, 14], aggs=["mean", "std"])

# Calendar features
result = pts.calendar_features(df, features=["day_of_week", "month", "is_weekend"])

# Fourier features for seasonality
result = pts.fourier_features(df, period=24, order=3)

# Advanced features
result = pts.target_encode(df, col="category", target="y")
result = pts.holiday_features(df, country="US")
result = pts.interaction_features(df, columns=["feat_a", "feat_b"])
result = pts.time_embeddings(df, components=["hour", "day_of_week"])
```

## Target transforms

All transforms are group-aware and invertible.

```python
# Log transform
transformed = pts.log_transform(df)
original = pts.inverse_log_transform(transformed)

# Box-Cox
transformed = pts.boxcox_transform(df, lam=0.5)
original = pts.inverse_boxcox_transform(transformed, lam=0.5)

# Differencing
diffed = pts.difference(df, order=1, period=24)
original = pts.undifference(diffed)
```

## Validation strategies

```python
# Expanding window cross-validation
splits = pts.expanding_window_cv(df, n_splits=5, horizon=12)

# Sliding window cross-validation
splits = pts.sliding_window_cv(df, train_size=100, horizon=12, step=12)

# Rolling origin CV
splits = pts.rolling_origin_cv(df, initial=100, horizon=12, fixed=False)
```

## Residual diagnostics

```python
acf_result = pts.acf(residuals, n_lags=40)
pacf_result = pts.pacf(residuals, n_lags=20)
lb_result = pts.ljung_box(residuals, lags=20)
```

## Further reading

- **Notebook 01**: [Data wrangling & exploration](https://github.com/drumtorben/polars-ts/blob/main/notebooks/01_data_wrangling_and_exploration.ipynb)
- **Notebook 02**: [Feature engineering & transforms](https://github.com/drumtorben/polars-ts/blob/main/notebooks/02_feature_engineering_transforms.ipynb)
