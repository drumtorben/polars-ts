"""Group-aware outlier detection and treatment for time series. Closes #61."""

from __future__ import annotations

import polars as pl


def detect_outliers(
    df: pl.DataFrame,
    target_col: str = "y",
    method: str = "zscore",
    id_col: str = "unique_id",
    time_col: str = "ds",
    threshold: float = 3.0,
    window: int | None = None,
) -> pl.DataFrame:
    """Detect outliers in a time series DataFrame.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to check for outliers.
    method
        Detection method: ``"zscore"``, ``"iqr"``, ``"hampel"``, or
        ``"rolling_zscore"``.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    threshold
        Sensitivity parameter. For zscore: number of std deviations
        (default 3). For IQR: multiplier (default 1.5 when
        ``threshold=1.5``).
    window
        Rolling window size (required for ``"rolling_zscore"`` and
        ``"hampel"``).

    Returns
    -------
    pl.DataFrame
        Original DataFrame with boolean ``is_outlier`` column appended.

    """
    valid_methods = {"zscore", "iqr", "hampel", "rolling_zscore"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid_methods)}")
    if method in ("rolling_zscore", "hampel") and window is None:
        raise ValueError(f"window is required for method={method!r}")

    sorted_df = df.sort(id_col, time_col)

    if method == "zscore":
        mean = pl.col(target_col).mean().over(id_col)
        std = pl.col(target_col).std().over(id_col)
        # When std is 0 (constant series), no values are outliers
        z = ((pl.col(target_col) - mean) / std).abs()
        return sorted_df.with_columns(z.fill_nan(0.0).gt(threshold).alias("is_outlier"))

    if method == "iqr":
        q1 = pl.col(target_col).quantile(0.25).over(id_col)
        q3 = pl.col(target_col).quantile(0.75).over(id_col)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        is_out = (pl.col(target_col) < lower) | (pl.col(target_col) > upper)
        return sorted_df.with_columns(is_out.alias("is_outlier"))

    if method == "hampel":
        assert window is not None
        import numpy as np

        result_frames: list[pl.DataFrame] = []
        for _gid, group_df in sorted_df.group_by(id_col, maintain_order=True):
            vals = np.array(group_df[target_col].to_list(), dtype=np.float64)
            n_vals = len(vals)
            is_out = [False] * n_vals
            half = window // 2
            for i in range(n_vals):
                lo = max(0, i - half)
                hi = min(n_vals, i + half + 1)
                win = vals[lo:hi]
                med = float(np.median(win))
                mad = float(np.median(np.abs(win - med)))
                if mad > 0 and abs(vals[i] - med) > threshold * 1.4826 * mad:
                    is_out[i] = True
            result_frames.append(group_df.with_columns(pl.Series("is_outlier", is_out)))
        return pl.concat(result_frames)

    if method == "rolling_zscore":
        assert window is not None
        rmean = pl.col(target_col).rolling_mean(window_size=window).over(id_col)
        rstd = pl.col(target_col).rolling_std(window_size=window).over(id_col)
        z = ((pl.col(target_col) - rmean) / rstd).abs()
        return sorted_df.with_columns(z.fill_null(0.0).gt(threshold).alias("is_outlier"))

    return sorted_df  # pragma: no cover


def treat_outliers(
    df: pl.DataFrame,
    target_col: str = "y",
    method: str = "zscore",
    replacement: str = "clip",
    id_col: str = "unique_id",
    time_col: str = "ds",
    threshold: float = 3.0,
    window: int | None = None,
) -> pl.DataFrame:
    """Detect and replace outliers in a time series DataFrame.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to treat.
    method
        Detection method (same as :func:`detect_outliers`).
    replacement
        How to replace outliers: ``"clip"`` (winsorize), ``"median"``
        (group median), ``"interpolate"`` (linear), or ``"null"``.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps.
    threshold
        Detection threshold.
    window
        Rolling window size for rolling methods.

    Returns
    -------
    pl.DataFrame
        DataFrame with outliers replaced.

    """
    valid_replacements = {"clip", "median", "interpolate", "null"}
    if replacement not in valid_replacements:
        raise ValueError(f"Unknown replacement {replacement!r}. Choose from {sorted(valid_replacements)}")

    detected = detect_outliers(df, target_col, method, id_col, time_col, threshold, window)

    if replacement == "null":
        result = detected.with_columns(
            pl.when(pl.col("is_outlier")).then(None).otherwise(pl.col(target_col)).alias(target_col)
        )
        return result.drop("is_outlier")

    if replacement == "clip":
        # Clip to detection boundaries per group
        if method == "iqr":
            q1 = pl.col(target_col).quantile(0.25).over(id_col)
            q3 = pl.col(target_col).quantile(0.75).over(id_col)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        else:
            mean = pl.col(target_col).mean().over(id_col)
            std = pl.col(target_col).std().over(id_col)
            lower = mean - threshold * std
            upper = mean + threshold * std
        result = detected.with_columns(pl.col(target_col).clip(lower, upper).alias(target_col))
        return result.drop("is_outlier")

    if replacement == "median":
        med = pl.col(target_col).median().over(id_col)
        result = detected.with_columns(
            pl.when(pl.col("is_outlier")).then(med).otherwise(pl.col(target_col)).alias(target_col)
        )
        return result.drop("is_outlier")

    if replacement == "interpolate":
        result = detected.with_columns(
            pl.when(pl.col("is_outlier")).then(None).otherwise(pl.col(target_col)).alias(target_col)
        )
        result = result.with_columns(pl.col(target_col).interpolate().over(id_col))
        return result.drop("is_outlier")

    return detected.drop("is_outlier")  # pragma: no cover
