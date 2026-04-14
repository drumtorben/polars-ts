"""Unified entry point for pairwise distance computation."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl

from polars_ts_rs.polars_ts_rs import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_dtw_multi,
    compute_pairwise_erp,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_msm_multi,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)

_UNIVARIATE_METHODS = {
    "dtw", "ddtw", "wdtw", "msm", "erp", "lcss", "twe",
}

_MULTIVARIATE_METHODS = {
    "dtw_multi", "msm_multi",
}

_ALL_METHODS = _UNIVARIATE_METHODS | _MULTIVARIATE_METHODS


def compute_pairwise_distance(
    input1: pl.DataFrame,
    input2: pl.DataFrame,
    method: Literal[
        "dtw", "ddtw", "wdtw", "msm", "erp", "lcss", "twe",
        "dtw_multi", "msm_multi",
    ] = "dtw",
    **kwargs: Any,
) -> pl.DataFrame:
    """Compute pairwise distance between time series using the specified method.

    This is the unified entry point for all distance metrics. It dispatches
    to the appropriate Rust implementation based on the ``method`` parameter.

    Args:
        input1: DataFrame with columns ``unique_id`` and ``y`` (univariate)
            or ``unique_id`` and multiple value columns (multivariate).
        input2: DataFrame with the same schema as ``input1``.
        method: The distance metric to use. One of:

            **Univariate:**

            - ``dtw`` — Dynamic Time Warping (default). Accepts ``dtw_method``
              (``standard``, ``sakoe_chiba``, ``itakura``, ``fast``) and ``param``.
            - ``ddtw`` — Derivative DTW. No extra parameters.
            - ``wdtw`` — Weighted DTW. Accepts ``g`` (weight penalty, default 0.05).
            - ``msm`` — Move-Split-Merge. Accepts ``c`` (cost, default 1.0).
            - ``erp`` — Edit Distance with Real Penalty. Accepts ``g`` (gap value, default 0.0).
            - ``lcss`` — Longest Common Subsequence. Accepts ``epsilon`` (threshold, default 1.0).
            - ``twe`` — Time Warp Edit Distance. Accepts ``nu`` (stiffness, default 0.001)
              and ``lambda_`` (edit penalty, default 1.0).

            **Multivariate:**

            - ``dtw_multi`` — Multivariate DTW. Accepts ``metric`` (``manhattan`` or ``euclidean``).
            - ``msm_multi`` — Multivariate MSM. Accepts ``c`` (cost, default 1.0).

        **kwargs: Method-specific parameters (see above).

    Returns:
        A DataFrame with columns ``unique_id_1``, ``unique_id_2``, and the distance column.

    Raises:
        ValueError: If an unknown method or unexpected keyword argument is passed.

    Examples:
        >>> compute_pairwise_distance(df, df, method="dtw")
        >>> compute_pairwise_distance(df, df, method="lcss", epsilon=0.5)
        >>> compute_pairwise_distance(df, df, method="dtw", dtw_method="sakoe_chiba", param=3.0)

    """
    if method not in _ALL_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {sorted(_ALL_METHODS)}"
        )

    if method == "dtw":
        _check_kwargs(kwargs, {"dtw_method", "param"}, method)
        dtw_kw: dict[str, Any] = {}
        if "dtw_method" in kwargs:
            dtw_kw["method"] = kwargs["dtw_method"]
        if "param" in kwargs:
            dtw_kw["param"] = kwargs["param"]
        return compute_pairwise_dtw(input1, input2, **dtw_kw)

    if method == "ddtw":
        _check_kwargs(kwargs, set(), method)
        return compute_pairwise_ddtw(input1, input2)

    if method == "wdtw":
        _check_kwargs(kwargs, {"g"}, method)
        return compute_pairwise_wdtw(input1, input2, **_pick(kwargs, "g"))

    if method == "msm":
        _check_kwargs(kwargs, {"c"}, method)
        return compute_pairwise_msm(input1, input2, **_pick(kwargs, "c"))

    if method == "erp":
        _check_kwargs(kwargs, {"g"}, method)
        return compute_pairwise_erp(input1, input2, **_pick(kwargs, "g"))

    if method == "lcss":
        _check_kwargs(kwargs, {"epsilon"}, method)
        return compute_pairwise_lcss(input1, input2, **_pick(kwargs, "epsilon"))

    if method == "twe":
        _check_kwargs(kwargs, {"nu", "lambda_"}, method)
        twe_kw: dict[str, Any] = {}
        if "nu" in kwargs:
            twe_kw["nu"] = kwargs["nu"]
        if "lambda_" in kwargs:
            # Rust param is named "lambda" which is a Python reserved word
            twe_kw["lambda"] = kwargs["lambda_"]
        return compute_pairwise_twe(input1, input2, **twe_kw)

    if method == "dtw_multi":
        _check_kwargs(kwargs, {"metric"}, method)
        return compute_pairwise_dtw_multi(input1, input2, **_pick(kwargs, "metric"))

    if method == "msm_multi":
        _check_kwargs(kwargs, {"c"}, method)
        return compute_pairwise_msm_multi(input1, input2, **_pick(kwargs, "c"))

    # unreachable due to the check above, but keeps mypy happy
    raise ValueError(f"Unknown method {method!r}")


def _pick(kwargs: dict, *keys: str) -> dict[str, Any]:
    """Return a dict with only the keys that are present in kwargs."""
    return {k: kwargs[k] for k in keys if k in kwargs}


def _check_kwargs(kwargs: dict, valid: set[str], method: str) -> None:
    unexpected = set(kwargs) - valid
    if unexpected:
        raise ValueError(
            f"Unexpected keyword argument(s) {sorted(unexpected)} for method {method!r}. "
            f"Valid options: {sorted(valid) if valid else '(none)'}"
        )
