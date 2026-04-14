"""Shared distance dispatch utilities for clustering and classification."""

from __future__ import annotations

from typing import Any

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

_DISTANCE_FUNCS = {
    "dtw": compute_pairwise_dtw,
    "ddtw": compute_pairwise_ddtw,
    "wdtw": compute_pairwise_wdtw,
    "msm": compute_pairwise_msm,
    "erp": compute_pairwise_erp,
    "lcss": compute_pairwise_lcss,
    "twe": compute_pairwise_twe,
    "dtw_multi": compute_pairwise_dtw_multi,
    "msm_multi": compute_pairwise_msm_multi,
}

_VALID_KWARGS = {
    "dtw": {"method", "param"},
    "ddtw": set(),
    "wdtw": {"g"},
    "msm": {"c"},
    "erp": {"g"},
    "lcss": {"epsilon"},
    "twe": {"nu", "lambda_"},
    "dtw_multi": {"metric"},
    "msm_multi": {"c"},
}


def compute_distances(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    method: str = "dtw",
    **kwargs: Any,
) -> pl.DataFrame:
    """Compute pairwise distances using the specified method."""
    if method not in _DISTANCE_FUNCS:
        raise ValueError(f"Unknown distance method {method!r}. Valid: {sorted(_DISTANCE_FUNCS)}")
    valid = _VALID_KWARGS[method]
    unexpected = set(kwargs) - valid
    if unexpected:
        raise ValueError(f"Unexpected kwargs {sorted(unexpected)} for method {method!r}")
    return _DISTANCE_FUNCS[method](df1, df2, **kwargs)


def pairwise_to_dict(df: pl.DataFrame) -> dict[tuple[str, str], float]:
    """Convert pairwise result DataFrame to {(id1, id2): distance} dict.

    The dict is symmetric: both (a, b) and (b, a) are stored.
    """
    rows = df.to_dicts()
    dist_col = [c for c in df.columns if c not in ("id_1", "id_2")][0]
    result: dict[tuple[str, str], float] = {}
    for r in rows:
        a, b = str(r["id_1"]), str(r["id_2"])
        d = r[dist_col]
        result[(a, b)] = d
        result[(b, a)] = d
    return result
