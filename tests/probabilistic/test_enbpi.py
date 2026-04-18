"""Tests for EnbPI (Ensemble Batch Prediction Intervals)."""

from datetime import date

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402

from polars_ts.probabilistic.conformal import EnbPI  # noqa: E402


def _make_linear_df() -> pl.DataFrame:
    n = 30
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) for i in range(n)] + [float(2 * i) for i in range(n)],
        }
    )


class TestEnbPI:
    def test_fit_predict_basic(self):
        df = _make_linear_df()
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2], coverage=0.9)
        enbpi.fit(df)
        result = enbpi.predict(df, h=3)

        assert "y_hat" in result.columns
        assert "y_hat_lower" in result.columns
        assert "y_hat_upper" in result.columns
        assert len(result) == 6  # 3 steps × 2 series

    def test_intervals_contain_point(self):
        df = _make_linear_df()
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2])
        enbpi.fit(df)
        result = enbpi.predict(df, h=2)

        for row in result.iter_rows(named=True):
            assert row["y_hat_lower"] <= row["y_hat"]
            assert row["y_hat"] <= row["y_hat_upper"]

    def test_update_changes_residuals(self):
        df = _make_linear_df()
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2])
        enbpi.fit(df)

        n_resids_before = len(enbpi.residuals_.get("A", []))

        new_obs = pl.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": [date(2024, 1, 31), date(2024, 2, 1)],
                "y": [30.0, 31.0],
                "y_hat": [29.5, 30.5],
            }
        )
        enbpi.update(new_obs)

        n_resids_after = len(enbpi.residuals_.get("A", []))
        assert n_resids_after == n_resids_before + 2

    def test_predict_before_fit(self):
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=5, lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            enbpi.predict(_make_linear_df(), h=1)

    def test_update_before_fit(self):
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=5, lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            enbpi.update(pl.DataFrame({"unique_id": ["A"], "ds": [date(2024, 1, 1)], "y": [1.0], "y_hat": [1.0]}))

    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="coverage"):
            EnbPI(lambda: LinearRegression(), coverage=0.0)

    def test_invalid_n_bootstraps(self):
        with pytest.raises(ValueError, match="n_bootstraps"):
            EnbPI(lambda: LinearRegression(), n_bootstraps=0)

    def test_multiple_series(self):
        df = _make_linear_df()
        enbpi = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2])
        enbpi.fit(df)
        result = enbpi.predict(df, h=1)

        assert len(result.filter(pl.col("unique_id") == "A")) == 1
        assert len(result.filter(pl.col("unique_id") == "B")) == 1

    def test_wider_coverage_wider_intervals(self):
        df = _make_linear_df()
        enbpi_narrow = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2], coverage=0.5)
        enbpi_narrow.fit(df)
        narrow = enbpi_narrow.predict(df, h=1)

        enbpi_wide = EnbPI(lambda: LinearRegression(), n_bootstraps=10, lags=[1, 2], coverage=0.95)
        enbpi_wide.fit(df)
        wide = enbpi_wide.predict(df, h=1)

        narrow_w = (narrow["y_hat_upper"] - narrow["y_hat_lower"]).mean()
        wide_w = (wide["y_hat_upper"] - wide["y_hat_lower"]).mean()
        assert wide_w >= narrow_w


def test_top_level_import():
    import polars_ts

    assert polars_ts.EnbPI is EnbPI


def test_submodule_import():
    from polars_ts.probabilistic import EnbPI as E

    assert callable(E)
