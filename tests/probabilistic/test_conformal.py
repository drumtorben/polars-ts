"""Tests for conformal prediction intervals."""

from datetime import date

import polars as pl
import pytest

from polars_ts.probabilistic.conformal import EnbPI, conformal_interval

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402

# ---------- conformal_interval ----------


def _make_cal_residuals(n: int = 50, residual_scale: float = 1.0) -> pl.DataFrame:
    """Calibration residuals: absolute values from a normal-ish distribution."""
    import numpy as np

    rng = np.random.default_rng(42)
    resids = np.abs(rng.normal(0, residual_scale, size=n))
    return pl.DataFrame({"residual": resids.tolist()})


def _make_predictions(n: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ds": [date(2024, 1, i + 1) for i in range(n)],
            "y_hat": [float(i) for i in range(n)],
        }
    )


class TestConformalInterval:
    def test_basic_output_columns(self):
        cal = _make_cal_residuals()
        pred = _make_predictions()
        result = conformal_interval(cal, pred, coverage=0.9)

        assert "y_hat_lower" in result.columns
        assert "y_hat_upper" in result.columns
        assert len(result) == len(pred)

    def test_interval_width_increases_with_coverage(self):
        cal = _make_cal_residuals()
        pred = _make_predictions()

        narrow = conformal_interval(cal, pred, coverage=0.5)
        wide = conformal_interval(cal, pred, coverage=0.95)

        narrow_width = (narrow["y_hat_upper"] - narrow["y_hat_lower"]).mean()
        wide_width = (wide["y_hat_upper"] - wide["y_hat_lower"]).mean()
        assert wide_width > narrow_width

    def test_symmetric_intervals(self):
        cal = _make_cal_residuals()
        pred = _make_predictions()
        result = conformal_interval(cal, pred, coverage=0.9, symmetric=True)

        lower_dist = result["y_hat"] - result["y_hat_lower"]
        upper_dist = result["y_hat_upper"] - result["y_hat"]
        for lo, hi in zip(lower_dist.to_list(), upper_dist.to_list(), strict=False):
            assert lo == pytest.approx(hi)

    def test_per_group_intervals(self):
        """Different groups can get different interval widths."""
        import numpy as np

        rng = np.random.default_rng(42)
        cal = pl.DataFrame(
            {
                "unique_id": ["A"] * 30 + ["B"] * 30,
                "residual": np.abs(rng.normal(0, 1.0, 30)).tolist() + np.abs(rng.normal(0, 5.0, 30)).tolist(),
            }
        )
        pred = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 5,
                "y_hat": [1.0] * 5 + [1.0] * 5,
            }
        )
        result = conformal_interval(cal, pred, coverage=0.9, id_col="unique_id")

        a_width = result.filter(pl.col("unique_id") == "A")["y_hat_upper"][0] - result.filter(
            pl.col("unique_id") == "A"
        )["y_hat_lower"][0]
        b_width = result.filter(pl.col("unique_id") == "B")["y_hat_upper"][0] - result.filter(
            pl.col("unique_id") == "B"
        )["y_hat_lower"][0]
        # Group B has much larger residuals
        assert b_width > a_width

    def test_zero_residuals(self):
        cal = pl.DataFrame({"residual": [0.0] * 20})
        pred = _make_predictions()
        result = conformal_interval(cal, pred, coverage=0.9)

        for row in result.iter_rows(named=True):
            assert row["y_hat_lower"] == pytest.approx(row["y_hat"])
            assert row["y_hat_upper"] == pytest.approx(row["y_hat"])

    def test_asymmetric_intervals(self):
        """Asymmetric mode uses signed residuals."""
        import numpy as np

        rng = np.random.default_rng(42)
        # Skewed residuals: mostly positive (model under-predicts)
        signed_resids = rng.exponential(1.0, size=100) - 0.5
        cal = pl.DataFrame({"residual": signed_resids.tolist()})
        pred = _make_predictions()
        result = conformal_interval(cal, pred, coverage=0.9, symmetric=False)

        # Lower and upper should NOT be equidistant from y_hat
        lower_dist = (result["y_hat"] - result["y_hat_lower"]).to_list()
        upper_dist = (result["y_hat_upper"] - result["y_hat"]).to_list()
        # At least some should differ
        assert not all(abs(lo - hi) < 1e-10 for lo, hi in zip(lower_dist, upper_dist, strict=False))

    def test_invalid_coverage(self):
        cal = _make_cal_residuals()
        pred = _make_predictions()
        with pytest.raises(ValueError, match="coverage"):
            conformal_interval(cal, pred, coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            conformal_interval(cal, pred, coverage=1.0)

    def test_missing_residual_col(self):
        cal = pl.DataFrame({"wrong_col": [1.0, 2.0]})
        pred = _make_predictions()
        with pytest.raises(ValueError, match="residual"):
            conformal_interval(cal, pred)

    def test_missing_predicted_col(self):
        cal = _make_cal_residuals()
        pred = pl.DataFrame({"wrong_col": [1.0, 2.0]})
        with pytest.raises(ValueError, match="y_hat"):
            conformal_interval(cal, pred)


# ---------- EnbPI ----------


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

        # Simulate new observations
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


# ---------- top-level imports ----------


def test_top_level_imports():
    import polars_ts

    assert polars_ts.conformal_interval is conformal_interval
    assert polars_ts.EnbPI is EnbPI


def test_submodule_imports():
    from polars_ts.probabilistic import EnbPI as E
    from polars_ts.probabilistic import conformal_interval as ci

    assert callable(ci)
    assert callable(E)


def test_metrics_namespace():
    """conformal_interval accessible via df.pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401 — registers .pts namespace

    cal = _make_cal_residuals()
    pred = _make_predictions()
    result = pred.pts.conformal_interval(cal, coverage=0.9)
    assert "y_hat_lower" in result.columns
    assert "y_hat_upper" in result.columns
