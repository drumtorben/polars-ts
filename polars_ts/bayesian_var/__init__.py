"""Bayesian Vector Autoregression (BVAR) for multivariate time series."""

from polars_ts.bayesian_var.model import (
    BayesianVAR,
    InferenceMethod,
    PriorType,
    bayesian_var,
)

# Internal helpers re-exported for backward compatibility with the pre-split
# single-module import paths. They are intentionally kept out of __all__ since
# they are private API; import them directly from the submodules instead.
from polars_ts.bayesian_var.model import _build_var_matrices as _build_var_matrices
from polars_ts.bayesian_var.priors import (
    MinnesotaPrior,
    NormalWishartPrior,
)
from polars_ts.bayesian_var.priors import _estimate_sigma_from_ar as _estimate_sigma_from_ar
from polars_ts.bayesian_var.priors import _minnesota_prior_precision as _minnesota_prior_precision
from polars_ts.bayesian_var.results import BayesianVARResult

__all__ = [
    "BayesianVAR",
    "BayesianVARResult",
    "InferenceMethod",
    "MinnesotaPrior",
    "NormalWishartPrior",
    "PriorType",
    "bayesian_var",
]
