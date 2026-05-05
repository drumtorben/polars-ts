# Bayesian Methods

polars-ts includes a full suite of Bayesian time series models — from linear state-space models to fully nonlinear particle filters.

## Kalman Filter & Smoother

Linear state-space model with exact inference. The RTS smoother provides optimally smoothed state estimates.

```python
import polars_ts as pts

kf = pts.KalmanFilter()
kf.fit(df)
smoothed = kf.smooth(df)
forecast = kf.predict(df, h=12)
```

## Bayesian Structural Time Series (BSTS)

Decomposes series into local level/trend, seasonality, and regression components with Bayesian priors.

```python
bsts = pts.BSTSModel(seasonal_period=12)
bsts.fit(df)
forecast = bsts.predict(df, h=12)
```

## Bayesian Exponential Smoothing

Exponential smoothing with priors over smoothing parameters and posterior predictive forecasting.

```python
bets = pts.BayesianETS()
bets.fit(df)
forecast = bets.predict(df, h=12)
```

## Bayesian VAR

Vector autoregression with Minnesota or Normal-Wishart priors for multivariate forecasting.

```python
bvar = pts.BayesianVAR(n_lags=4, prior="minnesota")
bvar.fit(df)
forecast = bvar.predict(df, h=8)
```

## Gaussian Process Regression

Non-parametric Bayesian forecasting with temporal kernels.

```python
gp = pts.GPForecaster(kernel="rbf")
gp.fit(df)
forecast = gp.predict(df, h=12)
```

## Unscented & Ensemble Kalman Filters

Nonlinear state-space models via sigma-point and ensemble approximations.

## Particle Filter / SMC

Fully nonlinear, non-Gaussian state-space models via Sequential Monte Carlo.

## MCMC Wrapper

Thin adapter for NumPyro and PyMC backends for custom Bayesian models.

## Bayesian Anomaly Scoring

Posterior predictive p-values and Bayes factors for principled anomaly detection.

```python
detector = pts.BayesianAnomalyDetector(warmup=50)
result = detector.fit_detect(df)
# result has columns: p_value, bayes_factor, is_anomaly
```
