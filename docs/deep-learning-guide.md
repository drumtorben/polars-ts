# Deep Learning & Foundation Models

polars-ts provides native deep learning forecasters, LLM-based adapters, and foundation model inference — all with the same `fit`/`predict` API.

## Native DL Forecasters

### N-BEATS

```python
import polars_ts as pts

model = pts.NBEATSForecaster(h=12, input_size=36, max_epochs=100)
model.fit(train_df)
forecasts = model.predict(train_df)
```

### PatchTST

Patch-based transformer for time series (Nie et al., ICLR 2023).

```python
model = pts.PatchTSTForecaster(h=12, input_size=64, patch_len=16, max_epochs=100)
model.fit(train_df)
forecasts = model.predict(train_df)
```

## Multivariate DL Forecasters

### MultivariatePatchTST

Channel-mixing PatchTST that jointly forecasts correlated variates.

```python
model = pts.MultivariatePatchTST(
    h=12, input_size=32, patch_len=8,
    target_cols=["price", "volume", "sentiment"],
    max_epochs=50,
)
model.fit(df)
result = model.predict(df)
# Columns: unique_id, ds, price_hat, volume_hat, sentiment_hat
```

### iTransformer

Inverted transformer (Liu et al., 2024) — treats each variate as a token for cross-variate attention.

```python
model = pts.iTransformerForecaster(
    h=12, input_size=30,
    target_cols=["price", "volume", "sentiment"],
    max_epochs=50,
)
model.fit(df)
result = model.predict(df)
```

## LLM-Based Forecasting

### Time-LLM

Patch embedding → cross-attention with learnable text prototypes → decode.

```python
model = pts.TimeLLMForecaster(h=12, input_size=36, max_epochs=50)
model.fit(df)
forecasts = model.predict(df)
```

### LLM-PS

Multi-scale CNN pattern extraction → semantic decoder.

```python
model = pts.LLMPSForecaster(
    h=12, input_size=36,
    kernel_sizes=[3, 5, 7],  # multi-scale patterns
    max_epochs=50,
)
model.fit(df)
forecasts = model.predict(df)
```

## Foundation Models (Zero-Shot)

Pre-trained models for zero-shot forecasting — no training required.

### Chronos

```python
forecaster = pts.ChronosForecaster(model_name="amazon/chronos-t5-small")
result = forecaster.predict(df, h=12)
# Includes prediction intervals: y_hat, y_hat_lower, y_hat_upper
```

### TimesFM

```python
forecaster = pts.TimesFMForecaster()
result = forecaster.predict(df, h=12)
```

### Moirai

```python
forecaster = pts.MoiraiForecaster(model_name="salesforce/moirai-1.0")
result = forecaster.predict(df, h=12)
```

## Deep Classifiers

### ROCKET / MiniROCKET

```python
clf = pts.RocketClassifier(n_kernels=1000)
clf.fit(train_df, labels)
predictions = clf.predict(test_df)
```

### InceptionTime / ResNet

```python
clf = pts.InceptionTimeClassifier(max_epochs=50)
clf.fit(train_df, labels)
predictions = clf.predict(test_df)
```
