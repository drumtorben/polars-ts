"""Group-aware missing value imputation for time series. Closes #60."""

from __future__ import annotations

import polars as pl


def impute(
    df: pl.DataFrame,
    target_col: str = "y",
    method: str = "forward_fill",
    id_col: str = "unique_id",
    time_col: str = "ds",
    season_length: int | None = None,
    add_indicator: bool = False,
) -> pl.DataFrame:
    """Impute missing values in a time series DataFrame.

    Parameters
    ----------
    df
        Input DataFrame.
    target_col
        Column to impute.
    method
        Imputation method: ``"forward_fill"``, ``"backward_fill"``,
        ``"linear"``, ``"mean"``, ``"median"``, or ``"seasonal"``.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.
    season_length
        Required when ``method="seasonal"``.
    add_indicator
        If ``True``, add a boolean ``{target_col}_imputed`` column.

    Returns
    -------
    pl.DataFrame
        DataFrame with missing values imputed.

    """
    valid_methods = {"forward_fill", "backward_fill", "linear", "mean", "median", "seasonal"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(valid_methods)}")
    if method == "seasonal" and season_length is None:
        raise ValueError("season_length is required for method='seasonal'")

    sorted_df = df.sort(id_col, time_col)

    if add_indicator:
        sorted_df = sorted_df.with_columns(pl.col(target_col).is_null().alias(f"{target_col}_imputed"))

    if method == "forward_fill":
        return sorted_df.with_columns(pl.col(target_col).forward_fill().over(id_col))

    if method == "backward_fill":
        return sorted_df.with_columns(pl.col(target_col).backward_fill().over(id_col))

    if method == "linear":
        return sorted_df.with_columns(pl.col(target_col).interpolate().over(id_col))

    if method == "mean":
        return sorted_df.with_columns(pl.col(target_col).fill_null(pl.col(target_col).mean().over(id_col)))

    if method == "median":
        return sorted_df.with_columns(pl.col(target_col).fill_null(pl.col(target_col).median().over(id_col)))

    if method == "seasonal":
        assert season_length is not None
        # Impute from same seasonal position within group
        result_frames: list[pl.DataFrame] = []
        for _gid, group_df in sorted_df.group_by(id_col, maintain_order=True):
            vals = group_df[target_col].to_list()
            for i, v in enumerate(vals):
                if v is None:
                    # Look for the nearest non-null value at the same seasonal position
                    for offset in range(1, len(vals) // season_length + 2):
                        back = i - offset * season_length
                        if back >= 0 and vals[back] is not None:
                            vals[i] = vals[back]
                            break
                        fwd = i + offset * season_length
                        if fwd < len(vals) and vals[fwd] is not None:
                            vals[i] = vals[fwd]
                            break
            result_frames.append(group_df.with_columns(pl.Series(target_col, vals)))
        return pl.concat(result_frames)

    return sorted_df  # pragma: no cover
