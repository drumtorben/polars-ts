# Forecasting Guide

polars-ts provides a complete forecasting stack — from simple baselines to ML pipelines, ensembles, and probabilistic intervals.

## Baseline models

Start with baselines to establish a benchmark before moving to complex models.

```python
import polars_ts as pts

naive = pts.naive_forecast(df, h=12)
seasonal = pts.seasonal_naive_forecast(df, h=12, season_length=24)
ma = pts.moving_average_forecast(df, h=12, window_size=7)
fft = pts.fft_forecast(df, h=12, n_harmonics=5)
```

## Exponential smoothing

Rust-accelerated implementations of SES, Holt, and Holt-Winters.

```python
ses = pts.ses_forecast(df, h=12, alpha=0.3)
holt = pts.holt_forecast(df, h=12, alpha=0.3, beta=0.1)
hw = pts.holt_winters_forecast(df, h=12, season_length=24, seasonal="additive")
```

## ARIMA / SARIMA

Two backends: manual order via `statsmodels`, or automatic selection via `statsforecast`.

```python
# Manual order
fitted = pts.arima_fit(df, order=(1, 1, 1))
forecast = pts.arima_forecast(fitted, h=12)

# Automatic
forecast = pts.auto_arima(df, h=12, season_length=12)
```

## ML forecast pipeline

`ForecastPipeline` wires up feature engineering, target transforms, and any sklearn-compatible model.

```python
from sklearn.ensemble import GradientBoostingRegressor

pipe = pts.ForecastPipeline(
    GradientBoostingRegressor(),
    lags=[1, 2, 7, 14],
    rolling_windows=[7, 14],
    calendar=["day_of_week", "month"],
    target_transform="log",
)
pipe.fit(train_df)
forecasts = pipe.predict(train_df, h=7)
```

## Multi-step strategies

```python
recursive = pts.RecursiveForecaster(model, lags=[1, 7])
recursive.fit(train_df)
preds = recursive.predict(train_df, h=14)

direct = pts.DirectForecaster(model, lags=[1, 7], h=14)
direct.fit(train_df)
preds = direct.predict(train_df)
```

## Global models

Train a single model across all series with optional series-identity encoding.

```python
gf = pts.GlobalForecaster(model, lags=[1, 7], id_encoding="ordinal")
gf.fit(train_df)
preds = gf.predict(train_df, h=7)
```

## Ensembles

```python
# Weighted combination
ens = pts.WeightedEnsemble(weights="inverse_error")
combined = ens.combine([forecast_a, forecast_b], validation_dfs=[val_a, val_b])

# Stacking meta-learner
stacker = pts.StackingForecaster(base_models=[model_a, model_b], meta_model=linear)
stacker.fit(train_df)
```

## Probabilistic forecasting

```python
# Quantile regression
qr = pts.QuantileRegressor(model, quantiles=[0.1, 0.5, 0.9])
qr.fit(train_df)
intervals = qr.predict(train_df, h=7)

# Conformal prediction intervals
result = pts.conformal_interval(cal_residuals, predictions, coverage=0.9)

# EnbPI — online adaptive intervals
enbpi = pts.EnbPI(model, n_bootstraps=100)
enbpi.fit(train_df)
intervals = enbpi.predict(train_df, h=7)
```

## Evaluation metrics

```python
from polars_ts import mae, rmse, mape, smape, mase, crps

print(f"MAE:  {mae(actuals, forecasts):.4f}")
print(f"RMSE: {rmse(actuals, forecasts):.4f}")
print(f"MAPE: {mape(actuals, forecasts):.4f}")
```

## Further reading

- **Notebook 03**: [Forecasting fundamentals](https://github.com/drumtorben/polars-ts/blob/main/notebooks/03_forecasting_fundamentals.ipynb)
- **Notebook 04**: [ML forecasting pipelines](https://github.com/drumtorben/polars-ts/blob/main/notebooks/04_ml_forecasting_pipelines.ipynb)
- **Notebook 05**: [Uncertainty & calibration](https://github.com/drumtorben/polars-ts/blob/main/notebooks/05_uncertainty_and_calibration.ipynb)
- **Notebook 09**: [Ensembles & reconciliation](https://github.com/drumtorben/polars-ts/blob/main/notebooks/09_ensembles_reconciliation.ipynb)
