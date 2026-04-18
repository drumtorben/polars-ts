"""Tests for conformal_interval (does not require sklearn)."""

from datetime import date

import polars as pl
import pytest

from polars_ts.probabilistic.conformal import conformal_interval


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

        a_width = (
            result.filter(pl.col("unique_id") == "A")["y_hat_upper"][0]
            - result.filter(pl.col("unique_id") == "A")["y_hat_lower"][0]
        )
        b_width = (
            result.filter(pl.col("unique_id") == "B")["y_hat_upper"][0]
            - result.filter(pl.col("unique_id") == "B")["y_hat_lower"][0]
        )
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
        signed_resids = rng.exponential(1.0, size=100) - 0.5
        cal = pl.DataFrame({"residual": signed_resids.tolist()})
        pred = _make_predictions()
        result = conformal_interval(cal, pred, coverage=0.9, symmetric=False)

        lower_dist = (result["y_hat"] - result["y_hat_lower"]).to_list()
        upper_dist = (result["y_hat_upper"] - result["y_hat"]).to_list()
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


def test_top_level_import():
    import polars_ts

    assert polars_ts.conformal_interval is conformal_interval


def test_submodule_import():
    from polars_ts.probabilistic import conformal_interval as ci

    assert callable(ci)


def test_metrics_namespace():
    """conformal_interval accessible via df.pts namespace."""
    from polars_ts.metrics import Metrics  # noqa: F401 — registers .pts namespace

    cal = _make_cal_residuals()
    pred = _make_predictions()
    result = pred.pts.conformal_interval(cal, coverage=0.9)
    assert "y_hat_lower" in result.columns
    assert "y_hat_upper" in result.columns
