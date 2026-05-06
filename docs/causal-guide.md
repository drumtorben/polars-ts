# Causal Inference

polars-ts provides causal inference methods for evaluating the impact of interventions on time series.

## CausalImpact

Bayesian structural time series counterfactual analysis. Estimates what would have happened without an intervention.

```python
import polars_ts as pts
from datetime import date

result = pts.causal_impact(
    df,
    intervention_date=date(2024, 3, 1),
    n_seasons=7,
)

print(result.summary())
# Pointwise and cumulative causal effects with credible intervals
```

### With covariates

```python
result = pts.causal_impact(
    df,
    intervention_date=date(2024, 3, 1),
    covariates=["weather", "demand"],
)
```

## Synthetic Control

Donor-pool weighted counterfactual estimation for panel data with treated and control units.

```python
sc = pts.SyntheticControl()
sc.fit(panel_df, treated_unit="unit_A", intervention_date=date(2024, 3, 1))

result = sc.estimate()
print(result.att)  # Average treatment effect on the treated
```

### Placebo tests

```python
# Run placebo tests to assess statistical significance
placebo = sc.placebo_test(n_permutations=100)
print(placebo.p_value)
```

## When to use which

| Method | Use case |
|--------|----------|
| **CausalImpact** | Single treated series, Bayesian counterfactual |
| **Synthetic Control** | Panel data with donor units, treatment effect estimation |
