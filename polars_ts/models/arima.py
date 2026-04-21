"""ARIMA / SARIMA wrappers for polars-ts.

* ``arima_fit`` / ``arima_forecast`` -- explicit order control via
  ``statsmodels.tsa.statespace.sarimax.SARIMAX``.
* ``auto_arima`` -- automatic order selection via
  ``statsforecast.models.AutoARIMA``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_freq(dates: pl.Series) -> timedelta:
    """Return the most common time delta in a sorted date/datetime column."""
    diffs = dates.diff().drop_nulls()
    if diffs.dtype == pl.Duration:
        # datetime column – diffs are already durations
        return diffs.mode().to_list()[0]
    # date column – diffs are i32 day counts
    day_count = diffs.cast(pl.Int64).mode().to_list()[0]
    return timedelta(days=int(day_count))


def _make_future_dates(
    last_date: Any,
    freq: timedelta,
    h: int,
) -> list[Any]:
    """Generate *h* future timestamps starting one step after *last_date*."""
    return [last_date + freq * (i + 1) for i in range(h)]


# ---------------------------------------------------------------------------
# auto_arima  (statsforecast backend)
# ---------------------------------------------------------------------------


def auto_arima(
    df: pl.DataFrame,
    h: int,
    season_length: int = 1,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Fit an AutoARIMA model per series and return *h*-step forecasts.

    Uses ``statsforecast.models.AutoARIMA`` under the hood.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]``.

    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA as _AutoARIMA
    except ImportError:
        raise ImportError(
            "statsforecast is required for auto_arima. " "Install it with: pip install polars-timeseries[forecast]"
        ) from None

    # Validate minimum series length per group
    group_lens = df.group_by(id_col).agg(pl.col(target_col).len().alias("_n"))
    too_short = group_lens.filter(pl.col("_n") < 3)
    if too_short.height > 0:
        short_ids = too_short[id_col].to_list()
        raise ValueError(f"Series too short for ARIMA estimation (need >= 3 observations): {short_ids}")

    # Build the statsforecast-compatible DataFrame
    sf_df = df.select(
        pl.col(id_col).alias("unique_id"),
        pl.col(time_col).alias("ds"),
        pl.col(target_col).cast(pl.Float64).alias("y"),
    ).to_pandas()

    sf = StatsForecast(
        models=[_AutoARIMA(season_length=season_length)],
        freq=1,
    )
    sf_result = sf.forecast(h=h, df=sf_df)

    # sf_result may have "unique_id" as index – reset it
    if "unique_id" not in sf_result.columns:
        sf_result = sf_result.reset_index()

    result = pl.from_pandas(sf_result)

    # Rename columns back to the caller's naming convention
    rename_map: dict[str, str] = {}
    if id_col != "unique_id":
        rename_map["unique_id"] = id_col
    if time_col != "ds":
        rename_map["ds"] = time_col
    # statsforecast names the prediction column after the model
    pred_col = [c for c in result.columns if c not in {"unique_id", "ds"}][0]
    rename_map[pred_col] = "y_hat"

    if rename_map:
        result = result.rename(rename_map)

    return result.select(id_col, time_col, "y_hat")


# ---------------------------------------------------------------------------
# arima_fit / arima_forecast  (statsmodels backend)
# ---------------------------------------------------------------------------


def arima_fit(
    df: pl.DataFrame,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    target_col: str = "y",
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> dict[str, Any]:
    """Fit a SARIMAX model per group and return a dict of fitted models.

    Parameters
    ----------
    df
        Long-format DataFrame with at least ``[id_col, time_col, target_col]``.
    order
        ``(p, d, q)`` for the non-seasonal component.
    seasonal_order
        ``(P, D, Q, s)`` for the seasonal component.  ``None`` means no
        seasonal component.

    Returns
    -------
    dict
        Mapping ``group_id -> fitted SARIMAXResults``.

    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError(
            "statsmodels is required for arima_fit/arima_forecast. " "Install it with: pip install statsmodels"
        ) from None

    fitted: dict[str, Any] = {}
    for key, group in df.group_by(id_col):
        group_id = str(key[0]) if isinstance(key, tuple) else str(key)
        group_sorted = group.sort(time_col)
        endog = group_sorted[target_col].to_numpy().astype(float)

        if len(endog) < 3:
            raise ValueError(
                f"Series {group_id!r} is too short for ARIMA estimation " f"(got {len(endog)} observations, need >= 3)."
            )

        kw: dict[str, Any] = {"order": order}
        if seasonal_order is not None:
            kw["seasonal_order"] = seasonal_order

        model = SARIMAX(endog, **kw)
        fitted[group_id] = model.fit(disp=False)

    return fitted


def arima_forecast(
    fitted: dict[str, Any],
    h: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pl.DataFrame:
    """Produce *h*-step-ahead forecasts from previously fitted models.

    Parameters
    ----------
    fitted
        Output of :func:`arima_fit`.
    h
        Forecast horizon.

    Returns
    -------
    pl.DataFrame
        Columns ``[id_col, time_col, "y_hat"]``.

    """
    parts: list[pl.DataFrame] = []
    for group_id, model_result in fitted.items():
        forecast_values = model_result.forecast(steps=h)
        part = pl.DataFrame(
            {
                id_col: [group_id] * h,
                time_col: list(range(1, h + 1)),
                "y_hat": forecast_values.tolist(),
            }
        )
        parts.append(part)

    return pl.concat(parts)
