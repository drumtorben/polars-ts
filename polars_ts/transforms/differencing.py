"""Differencing and undifferencing for time series target columns."""

from __future__ import annotations

import polars as pl


def difference(
    df: pl.DataFrame,
    order: int = 1,
    period: int = 1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Apply differencing to target column, replacing it with differenced values.

    Parameters
    ----------
    df
        Input DataFrame.
    order
        Number of times to difference (1 = first-order, 2 = second-order).
    period
        Seasonal period (1 = regular differencing, m = seasonal).
    target_col
        Column to difference.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` replaced by differenced values.
        Stores initial values in ``{target_col}_diff_initial`` (list column)
        for inversion. Null rows produced by differencing are dropped.

    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if period < 1:
        raise ValueError("period must be >= 1")

    init_col = f"{target_col}_diff_initial"
    if init_col in df.columns:
        raise ValueError(f"Column {init_col!r} already exists — differencing may have been applied already")

    sorted_df = df.sort(id_col, time_col)

    if sorted_df.is_empty():
        return sorted_df.with_columns(
            pl.Series(init_col, [], dtype=pl.List(pl.Float64)),
        )

    def _store_and_diff(group: pl.DataFrame) -> pl.DataFrame:
        vals = group[target_col].to_list()

        # Build intermediate series: intermediates[0] = original,
        # intermediates[k] = after k-th differencing pass.
        intermediates = [list(vals)]
        current = list(vals)
        for _ in range(order):
            new = []
            for i in range(len(current)):
                if i >= period and current[i] is not None and current[i - period] is not None:
                    new.append(current[i] - current[i - period])
                else:
                    new.append(None)
            current = new
            intermediates.append(current)

        # The surviving window starts at index `period * order`.
        # For undifferencing pass k (0-indexed), we need the `period` values
        # from intermediates[k] immediately before the surviving window:
        # indices `[survive_start - period, survive_start)`.
        survive_start = period * order
        prefix_start = survive_start - period
        all_prefixes: list[float] = []
        for k in range(order):
            all_prefixes.extend(intermediates[k][prefix_start:survive_start])

        # Count contiguous non-nulls from the tail
        non_null_count = 0
        for v in reversed(current):
            if v is None:
                break
            non_null_count += 1

        if non_null_count == 0:
            return group.head(0).with_columns(
                pl.Series(target_col, [], dtype=pl.Float64),
                pl.Series(init_col, [], dtype=pl.List(pl.Float64)),
            )

        diffed_vals = current[len(current) - non_null_count :]
        result = group.tail(non_null_count)
        return result.with_columns(
            pl.Series(target_col, diffed_vals),
            pl.Series(init_col, [all_prefixes] * non_null_count),
        )

    return sorted_df.group_by(id_col, maintain_order=True).map_groups(_store_and_diff)


def undifference(
    df: pl.DataFrame,
    order: int = 1,
    period: int = 1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Invert differencing, restoring target column to original scale.

    Parameters
    ----------
    df
        DataFrame with differenced target column and
        ``{target_col}_diff_initial`` metadata.
    order
        Number of differencing passes to invert.
    period
        Seasonal period used in differencing.
    target_col
        Column to restore.
    id_col
        Column identifying each time series.
    time_col
        Column with timestamps for ordering.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``target_col`` restored and metadata column dropped.

    """
    init_col = f"{target_col}_diff_initial"
    if init_col not in df.columns:
        raise ValueError(f"Column {init_col!r} not found — cannot invert without initial values")

    sorted_df = df.sort(id_col, time_col)

    def _undiff_group(group: pl.DataFrame) -> pl.DataFrame:
        all_prefixes = group[init_col][0].to_list()
        diffed = group[target_col].to_list()

        # Invert each differencing pass in reverse order.
        # Prefixes are stored as [pass0_prefix, pass1_prefix, ...],
        # each of length `period`. Undo in reverse: pass (order-1) first.
        current = diffed
        for pass_idx in range(order - 1, -1, -1):
            start = pass_idx * period
            prefix = all_prefixes[start : start + period]
            current = _cumsum_with_prefix(prefix, current, period)

        return group.with_columns(pl.Series(target_col, current))

    result = sorted_df.group_by(id_col, maintain_order=True).map_groups(_undiff_group)
    return result.drop(init_col)


def _cumsum_with_prefix(
    prefix: list[float],
    diffed: list[float],
    period: int,
) -> list[float]:
    """Reconstruct original values from differenced values and initial prefix."""
    full = prefix + diffed
    result = [0.0] * len(full)

    if period == 1:
        result[0] = full[0]
        for i in range(1, len(full)):
            result[i] = result[i - 1] + full[i]
    else:
        for s in range(period):
            indices = list(range(s, len(full), period))
            result[indices[0]] = full[indices[0]]
            for j in range(1, len(indices)):
                result[indices[j]] = result[indices[j - 1]] + full[indices[j]]

    # Remove prepended prefix, return only the diffed-length portion
    return result[period:]
