"""Shared inverse-transform helpers for pipeline and global model.

Canonical location for inverse_single and transform_buffer,
previously duplicated across pipeline.py and global_model.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def inverse_single(
    pred: float,
    target_transform: str | None,
    transform_state: dict[str, Any],
    orig_values: list[float],
) -> float:
    """Inverse-transform a single prediction to original scale."""
    if target_transform == "log":
        return float(np.expm1(pred))
    if target_transform == "boxcox":
        lam = transform_state.get("lam", 1.0)
        if lam == 0:
            return float(np.exp(pred))
        return float((pred * lam + 1) ** (1.0 / lam))
    if target_transform == "difference":
        period = transform_state.get("period", 1)
        if len(orig_values) >= period:
            return pred + orig_values[-period]
        return pred
    return pred


def transform_buffer(
    values: list[float],
    target_transform: str | None,
    transform_state: dict[str, Any],
) -> list[float]:
    """Transform raw values into the model's working space."""
    if target_transform == "log":
        return [float(np.log1p(v)) for v in values]
    if target_transform == "boxcox":
        lam = transform_state.get("lam", 1.0)
        if lam == 0:
            return [float(np.log(v)) for v in values]
        return [float((v**lam - 1) / lam) for v in values]
    if target_transform == "difference":
        period = transform_state.get("period", 1)
        order = transform_state.get("order", 1)
        result = list(values)
        for _ in range(order):
            result = [result[i] - result[i - period] if i >= period else float("nan") for i in range(len(result))]
        return [v for v in result if not (isinstance(v, float) and np.isnan(v))]
    return list(values)
