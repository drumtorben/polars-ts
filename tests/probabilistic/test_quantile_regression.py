"""Tests for quantile regression forecaster."""

from datetime import date

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

from polars_ts.probabilistic.quantile_regression import QuantileRegressor  # noqa: E402


def _make_linear_df() -> pl.DataFrame:
    """Two series with perfect linear trends: A = t, B = 2t."""
    n = 20
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) for i in range(n)] + [float(2 * i) for i in range(n)],
        }
    )


def _gbr_factory(q: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=50, max_depth=3, random_state=42)


class TestQuantileRegressor:
    def test_fit_predict_basic(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.1, 0.5, 0.9], lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=3)

        assert "y_hat" in result.columns
        assert "q_0.1" in result.columns
        assert "q_0.5" in result.columns
        assert "q_0.9" in result.columns
        assert len(result) == 6  # 3 steps × 2 series

    def test_quantile_ordering(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.1, 0.5, 0.9], lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=2)

        for row in result.iter_rows(named=True):
            assert row["q_0.1"] <= row["q_0.5"] + 1e-6
            assert row["q_0.5"] <= row["q_0.9"] + 1e-6

    def test_output_columns_match_quantiles(self):
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=quantiles, lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=1)

        for q in quantiles:
            assert f"q_{q}" in result.columns

    def test_median_equals_y_hat(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.1, 0.5, 0.9], lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=2)

        for row in result.iter_rows(named=True):
            assert row["y_hat"] == pytest.approx(row["q_0.5"])

    def test_multiple_series(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.1, 0.9], lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=2)

        a = result.filter(pl.col("unique_id") == "A")
        b = result.filter(pl.col("unique_id") == "B")
        assert len(a) == 2
        assert len(b) == 2

    def test_future_dates_correct(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.5], lags=[1])
        qr.fit(df)
        result = qr.predict(df, h=2)

        a = result.filter(pl.col("unique_id") == "A")
        assert a["ds"].to_list() == [date(2024, 1, 21), date(2024, 1, 22)]

    def test_predict_before_fit(self):
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.5], lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            qr.predict(_make_linear_df(), h=1)

    def test_invalid_quantiles_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuantileRegressor(_gbr_factory, quantiles=[], lags=[1])

    def test_invalid_quantiles_range(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            QuantileRegressor(_gbr_factory, quantiles=[0.0, 0.5], lags=[1])
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            QuantileRegressor(_gbr_factory, quantiles=[0.5, 1.0], lags=[1])

    def test_invalid_lags(self):
        with pytest.raises(ValueError, match="positive"):
            QuantileRegressor(_gbr_factory, quantiles=[0.5], lags=[])

    def test_invalid_horizon(self):
        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.5], lags=[1])
        qr.fit(df)
        with pytest.raises(ValueError, match="positive"):
            qr.predict(df, h=0)

    def test_crps_compatible(self):
        """Output can be passed directly to crps() metric."""
        from polars_ts.metrics.forecast import crps

        df = _make_linear_df()
        qr = QuantileRegressor(_gbr_factory, quantiles=[0.1, 0.5, 0.9], lags=[1, 2])
        qr.fit(df)
        result = qr.predict(df, h=1)

        # Add fake actuals for CRPS computation
        result_with_actuals = result.with_columns(pl.col("y_hat").alias("y"))
        score = crps(result_with_actuals)
        assert isinstance(score, float)
        assert score >= 0


def test_top_level_import():
    import polars_ts

    assert polars_ts.QuantileRegressor is QuantileRegressor


def test_submodule_import():
    from polars_ts.probabilistic import QuantileRegressor as QR

    assert callable(QR)
