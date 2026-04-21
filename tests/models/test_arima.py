"""Tests for polars_ts.models.arima (ARIMA / SARIMA wrappers)."""

from __future__ import annotations

import random
from datetime import date, timedelta

import polars as pl
import pytest


def _make_df(
    n: int = 50,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    series_ids: list[str] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a simple trending series with some noise."""
    rng = random.Random(seed)
    if series_ids is None:
        series_ids = ["A"]

    parts: list[pl.DataFrame] = []
    for sid in series_ids:
        noise = [rng.gauss(0, 1) for _ in range(n)]
        parts.append(
            pl.DataFrame(
                {
                    id_col: [sid] * n,
                    time_col: [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                    target_col: [10.0 + 0.5 * i + noise[i] for i in range(n)],
                }
            )
        )
    return pl.concat(parts)


# ---------------------------------------------------------------------------
# auto_arima tests (statsforecast backend)
# ---------------------------------------------------------------------------


class TestAutoArima:
    def test_auto_arima_basic(self) -> None:
        sf = pytest.importorskip("statsforecast")  # noqa: F841
        from polars_ts.models.arima import auto_arima

        df = _make_df(n=50)
        result = auto_arima(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert result.height == 3

    def test_auto_arima_multi_series(self) -> None:
        sf = pytest.importorskip("statsforecast")  # noqa: F841
        from polars_ts.models.arima import auto_arima

        df = _make_df(n=50, series_ids=["A", "B"])
        result = auto_arima(df, h=3)

        assert result.height == 6
        assert set(result["unique_id"].unique().to_list()) == {"A", "B"}

    def test_auto_arima_with_season_length(self) -> None:
        sf = pytest.importorskip("statsforecast")  # noqa: F841
        from polars_ts.models.arima import auto_arima

        df = _make_df(n=60)
        result = auto_arima(df, h=5, season_length=12)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert result.height == 5


# ---------------------------------------------------------------------------
# arima_fit / arima_forecast tests (statsmodels backend)
# ---------------------------------------------------------------------------


class TestArimaFitForecast:
    def test_arima_fit_forecast_roundtrip(self) -> None:
        sm = pytest.importorskip("statsmodels")  # noqa: F841
        from polars_ts.models.arima import arima_fit, arima_forecast

        df = _make_df(n=50)
        fitted = arima_fit(df)

        assert isinstance(fitted, dict)
        assert "A" in fitted

        result = arima_forecast(fitted, h=3)
        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert result.height == 3

    def test_arima_seasonal(self) -> None:
        sm = pytest.importorskip("statsmodels")  # noqa: F841
        from polars_ts.models.arima import arima_fit, arima_forecast

        df = _make_df(n=60)
        fitted = arima_fit(df, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
        result = arima_forecast(fitted, h=5)

        assert result.height == 5

    def test_arima_custom_columns(self) -> None:
        sm = pytest.importorskip("statsmodels")  # noqa: F841
        from polars_ts.models.arima import arima_fit, arima_forecast

        df = _make_df(
            n=50,
            id_col="series",
            time_col="timestamp",
            target_col="value",
        )
        fitted = arima_fit(
            df,
            target_col="value",
            id_col="series",
            time_col="timestamp",
        )

        assert isinstance(fitted, dict)

        result = arima_forecast(fitted, h=3, id_col="series", time_col="timestamp")
        assert result.columns == ["series", "timestamp", "y_hat"]
        assert result.height == 3

    def test_arima_short_series_raises(self) -> None:
        sm = pytest.importorskip("statsmodels")  # noqa: F841
        from polars_ts.models.arima import arima_fit

        df = _make_df(n=2)
        with pytest.raises(ValueError, match="too short"):
            arima_fit(df)


# ---------------------------------------------------------------------------
# Top-level import test
# ---------------------------------------------------------------------------


def test_top_level_import() -> None:
    import polars_ts

    assert hasattr(polars_ts, "auto_arima")
    assert callable(polars_ts.auto_arima)
