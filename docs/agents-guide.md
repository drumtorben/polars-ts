# Agentic Forecasting & Multi-Agent RL

polars-ts includes multi-agent frameworks for automated forecasting pipelines, portfolio optimization, and autonomous anomaly detection.

## TimeSeriesScientist

Automates the full forecasting workflow: data diagnostics → model selection → forecasting → reporting.

```python
import polars_ts as pts

scientist = pts.TimeSeriesScientist(horizon=7)
result = scientist.run(df)

print(result.report)        # Markdown report
print(result.predictions)   # DataFrame with y_hat
print(result.context.history)  # Agent decision log
```

### Pipeline stages

| Agent | Responsibility |
|-------|---------------|
| **CuratorAgent** | Data quality diagnostics, cleaning, regime-change lookback trimming |
| **PlannerAgent** | Model selection based on data characteristics |
| **ForecasterAgent** | Fit candidates, validate via MAE, build ensemble |
| **ReporterAgent** | Generate structured markdown report |

### With events and lookback trimming

```python
scientist = pts.TimeSeriesScientist(
    horizon=14,
    events=[{"date": "2024-06-01", "description": "Product launch"}],
    trim_lookback=True,  # Auto-trim based on regime change detection
)
result = scientist.run(df)
```

### Custom LLM backend

```python
class MyLLMBackend:
    def complete(self, prompt: str) -> str:
        return my_llm_api.chat(prompt)

scientist = pts.TimeSeriesScientist(horizon=7, backend=MyLLMBackend())
```

## Multi-Agent RL for Portfolio Optimization

Specialized agents collaborate for time series-driven portfolio decisions.

```python
orch = pts.MARLOrchestrator(window_size=20, risk_aversion=1.5)
result = orch.run(returns_matrix)

print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Total return: {result.total_return:.2%}")
print(result.weights_history)  # (n_steps, n_assets)
```

### From a price DataFrame

```python
result = orch.run_from_dataframe(price_df)
```

### Agents

| Agent | Method | Description |
|-------|--------|-------------|
| **RiskAgent** | `assess()` | Rolling volatility per asset |
| **ReturnAgent** | `predict()` | Exponentially weighted expected returns |
| **AllocationAgent** | `allocate()` | Risk-adjusted return → normalized weights |

## Autonomous Anomaly Detection

Multi-agent anomaly detection with consensus voting.

```python
orch = pts.AnomalyOrchestrator(window_size=20)
result = orch.run(time_series)

print(result.detections)     # Boolean array
print(result.agent_scores)   # Per-agent score arrays
```

### Detection agents

| Agent | Method | Description |
|-------|--------|-------------|
| **ZScoreAgent** | Z-score | Classic statistical detection |
| **RollingStdAgent** | Median-deviation | Robust rolling detection |
| **MADAgent** | MAD | Median Absolute Deviation (outlier-resistant) |
| **ConsensusAgent** | Voting | Majority, any, or weighted aggregation |
