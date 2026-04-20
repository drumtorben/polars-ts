"""Tests for stacking forecaster."""

from datetime import date

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402

from polars_ts.ensemble.stacking import StackingForecaster  # noqa: E402


def _make_forecast(values: list[float], series_id: str = "A") -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "unique_id": [series_id] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)],
            "y_hat": values,
        }
    )


def _make_actuals(values: list[float], series_id: str = "A") -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "unique_id": [series_id] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)],
            "y": values,
        }
    )


class TestStackingForecaster:
    def test_fit_predict_basic(self):
        # Two base models with CV predictions
        cv1 = _make_forecast([10.0, 20.0, 30.0, 40.0, 50.0])
        cv2 = _make_forecast([12.0, 22.0, 28.0, 42.0, 48.0])
        actuals = _make_actuals([11.0, 21.0, 29.0, 41.0, 49.0])

        sf = StackingForecaster(LinearRegression())
        sf.fit([cv1, cv2], actuals)

        test1 = _make_forecast([60.0])
        test2 = _make_forecast([58.0])
        result = sf.predict([test1, test2])

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 1

    def test_output_schema(self):
        cv1 = _make_forecast([1.0, 2.0, 3.0])
        cv2 = _make_forecast([2.0, 3.0, 4.0])
        actuals = _make_actuals([1.5, 2.5, 3.5])

        sf = StackingForecaster(LinearRegression())
        sf.fit([cv1, cv2], actuals)

        result = sf.predict([_make_forecast([5.0]), _make_forecast([6.0])])
        assert result.columns == ["unique_id", "ds", "y_hat"]

    def test_predict_before_fit(self):
        sf = StackingForecaster(LinearRegression())
        with pytest.raises(RuntimeError, match="fit"):
            sf.predict([_make_forecast([1.0]), _make_forecast([2.0])])

    def test_wrong_number_of_forecasts(self):
        cv1 = _make_forecast([1.0, 2.0, 3.0])
        cv2 = _make_forecast([2.0, 3.0, 4.0])
        actuals = _make_actuals([1.5, 2.5, 3.5])

        sf = StackingForecaster(LinearRegression())
        sf.fit([cv1, cv2], actuals)

        with pytest.raises(ValueError, match="2 forecasts"):
            sf.predict([_make_forecast([5.0])])

    def test_multiple_series(self):
        cv1 = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "ds": [date(2024, 1, i + 1) for i in range(3)] * 2,
                "y_hat": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )
        cv2 = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "ds": [date(2024, 1, i + 1) for i in range(3)] * 2,
                "y_hat": [12.0, 22.0, 28.0, 110.0, 210.0, 290.0],
            }
        )
        actuals = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "ds": [date(2024, 1, i + 1) for i in range(3)] * 2,
                "y": [11.0, 21.0, 29.0, 105.0, 205.0, 295.0],
            }
        )

        sf = StackingForecaster(LinearRegression())
        sf.fit([cv1, cv2], actuals)

        test1 = pl.DataFrame({"unique_id": ["A", "B"], "ds": [date(2024, 1, 4)] * 2, "y_hat": [40.0, 400.0]})
        test2 = pl.DataFrame({"unique_id": ["A", "B"], "ds": [date(2024, 1, 4)] * 2, "y_hat": [38.0, 390.0]})
        result = sf.predict([test1, test2])
        assert len(result) == 2

    def test_stacking_with_ridge(self):
        """Works with different meta-learners."""
        cv1 = _make_forecast([1.0, 2.0, 3.0, 4.0, 5.0])
        cv2 = _make_forecast([1.5, 2.5, 3.5, 4.5, 5.5])
        actuals = _make_actuals([1.2, 2.2, 3.2, 4.2, 5.2])

        sf = StackingForecaster(Ridge(alpha=0.1))
        sf.fit([cv1, cv2], actuals)
        result = sf.predict([_make_forecast([6.0]), _make_forecast([6.5])])
        assert len(result) == 1

    def test_stacking_beats_individual(self):
        """Stacking should outperform individual biased models."""
        from polars_ts.metrics.forecast import mae

        n = 20
        true_vals = [float(i) for i in range(n)]
        # Model A: consistently over-predicts by 2
        model_a_vals = [v + 2.0 for v in true_vals]
        # Model B: consistently under-predicts by 2
        model_b_vals = [v - 2.0 for v in true_vals]

        cv1 = _make_forecast(model_a_vals)
        cv2 = _make_forecast(model_b_vals)
        actuals = _make_actuals(true_vals)

        sf = StackingForecaster(LinearRegression())
        sf.fit([cv1, cv2], actuals)

        # Test set
        test_true = [float(i) for i in range(n, n + 5)]
        test1 = _make_forecast([v + 2.0 for v in test_true])
        test2 = _make_forecast([v - 2.0 for v in test_true])
        result = sf.predict([test1, test2])

        test_actuals = _make_actuals(test_true)
        combined = test_actuals.join(result, on=["unique_id", "ds"])
        stacking_mae = mae(combined, actual_col="y", predicted_col="y_hat")

        # Individual MAEs are both 2.0
        assert stacking_mae < 1.0  # Stacking should be much better

    def test_empty_cv_predictions(self):
        sf = StackingForecaster(LinearRegression())
        with pytest.raises(ValueError, match="non-empty"):
            sf.fit([], _make_actuals([1.0]))

    def test_single_model_rejected(self):
        sf = StackingForecaster(LinearRegression())
        with pytest.raises(ValueError, match="at least 2"):
            sf.fit([_make_forecast([1.0])], _make_actuals([1.0]))


def test_top_level_import():
    import polars_ts

    assert polars_ts.StackingForecaster is StackingForecaster


def test_submodule_import():
    from polars_ts.ensemble import StackingForecaster as SF

    assert callable(SF)
