"""CausalImpactResult dataclass — result container for causal impact analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polars_ts.bayesian.bsts import BSTSResult


@dataclass
class CausalImpactResult:
    """Result container for a CausalImpact analysis.

    All effect arrays have length equal to the post-intervention period.

    Attributes
    ----------
    point_effect
        Pointwise causal effect (observed - counterfactual).
    point_effect_lower
        Lower credible bound of the pointwise effect.
    point_effect_upper
        Upper credible bound of the pointwise effect.
    cumulative_effect
        Cumulative sum of pointwise effects.
    cumulative_effect_lower
        Conservative lower bound of cumulative effect (sum of
        pointwise lower bounds). This is wider than a proper
        posterior-based interval because it does not account for
        cross-timestep correlation.
    cumulative_effect_upper
        Conservative upper bound of cumulative effect.
    total_effect
        Sum of pointwise effects over the post period.
    total_effect_lower
        Lower credible bound of total effect.
    total_effect_upper
        Upper credible bound of total effect.
    relative_effect
        Total effect divided by sum of counterfactual.
    relative_effect_lower
        Lower credible bound of relative effect.
    relative_effect_upper
        Upper credible bound of relative effect.
    counterfactual
        Predicted counterfactual series for the post period.
    counterfactual_lower
        Lower credible bound of counterfactual.
    counterfactual_upper
        Upper credible bound of counterfactual.
    observed_post
        Observed values in the post period.
    bsts_result
        Underlying BSTS model result.
    pre_mape
        Mean absolute percentage error on the pre-period (diagnostic).
    pre_coverage
        Fraction of pre-period observations inside the credible interval.

    """

    point_effect: np.ndarray
    point_effect_lower: np.ndarray
    point_effect_upper: np.ndarray
    cumulative_effect: np.ndarray
    cumulative_effect_lower: np.ndarray
    cumulative_effect_upper: np.ndarray
    total_effect: float
    total_effect_lower: float
    total_effect_upper: float
    relative_effect: float
    relative_effect_lower: float
    relative_effect_upper: float
    counterfactual: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray
    observed_post: np.ndarray
    bsts_result: BSTSResult
    pre_mape: float
    pre_coverage: float
