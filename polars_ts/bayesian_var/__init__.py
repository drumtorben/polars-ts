"""Bayesian Vector Autoregression (BVAR) for multivariate time series."""

from polars_ts.bayesian_var.model import (
    BayesianVAR,
    InferenceMethod,
    PriorType,
    _build_var_matrices,
    bayesian_var,
)
from polars_ts.bayesian_var.priors import (
    MinnesotaPrior,
    NormalWishartPrior,
    _estimate_sigma_from_ar,
    _minnesota_prior_precision,
)
from polars_ts.bayesian_var.results import BayesianVARResult

__all__ = [
    "BayesianVAR",
    "BayesianVARResult",
    "InferenceMethod",
    "MinnesotaPrior",
    "NormalWishartPrior",
    "PriorType",
    "_build_var_matrices",
    "_estimate_sigma_from_ar",
    "_minnesota_prior_precision",
    "bayesian_var",
]
