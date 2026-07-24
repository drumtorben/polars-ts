"""Bayesian Exponential Smoothing (Bayesian ETS)."""

# Internal helpers re-exported for backward compatibility with the pre-split
# single-module import paths. They are intentionally kept out of __all__ since
# they are private API; import them directly from the submodule instead.
from polars_ts.models.bayesian_ets.inference import _forecast_from_params as _forecast_from_params
from polars_ts.models.bayesian_ets.inference import _holt_loglik as _holt_loglik
from polars_ts.models.bayesian_ets.inference import _hw_loglik as _hw_loglik
from polars_ts.models.bayesian_ets.inference import _log_posterior as _log_posterior
from polars_ts.models.bayesian_ets.inference import _map_estimate as _map_estimate
from polars_ts.models.bayesian_ets.inference import _mcmc_sample as _mcmc_sample
from polars_ts.models.bayesian_ets.inference import _pack_params as _pack_params
from polars_ts.models.bayesian_ets.inference import _ses_loglik as _ses_loglik
from polars_ts.models.bayesian_ets.inference import _unpack_params as _unpack_params
from polars_ts.models.bayesian_ets.model import BayesianETS, BayesianETSResult, bayesian_ets
from polars_ts.models.bayesian_ets.priors import ETSPriors, InferenceMethod, ModelType

__all__ = [
    "BayesianETS",
    "BayesianETSResult",
    "ETSPriors",
    "InferenceMethod",
    "ModelType",
    "bayesian_ets",
]
