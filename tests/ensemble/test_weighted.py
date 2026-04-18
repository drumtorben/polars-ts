"""Tests for weighted forecast ensemble."""

from datetime import date

import polars as pl
import pytest

from polars_ts.ensemble.weighted import WeightedEnsemble


def _make_forecast(values: list[float], series_id: str = "A") -> pl.DataFrame:
    n = len(values)
    return pl.DataFrame(
        {
            "unique_id": [series_id] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)],
            "y_hat": values,
        }
    )


class TestWeightedEnsemble:
    def test_equal_weights(self):
        f1 = _make_forecast([10.0, 20.0])
        f2 = _make_forecast([20.0, 40.0])
        ens = WeightedEnsemble(weights="equal")
        result = ens.combine([f1, f2])

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert result["y_hat"].to_list() == pytest.approx([15.0, 30.0])

    def test_manual_weights(self):
        f1 = _make_forecast([10.0, 20.0])
        f2 = _make_forecast([20.0, 40.0])
        ens = WeightedEnsemble(weights=[0.75, 0.25])
        result = ens.combine([f1, f2])

        # 0.75*10 + 0.25*20 = 12.5, 0.75*20 + 0.25*40 = 25.0
        assert result["y_hat"].to_list() == pytest.approx([12.5, 25.0])

    def test_manual_weights_normalized(self):
        """Weights don't need to sum to 1 — they get normalized."""
        f1 = _make_forecast([10.0])
        f2 = _make_forecast([20.0])
        ens = WeightedEnsemble(weights=[3.0, 1.0])
        result = ens.combine([f1, f2])

        # 0.75*10 + 0.25*20 = 12.5
        assert result["y_hat"][0] == pytest.approx(12.5)

    def test_inverse_error_weights(self):
        f1 = _make_forecast([10.0, 20.0])
        f2 = _make_forecast([20.0, 40.0])

        # Model 1 has MAE=1, Model 2 has MAE=2
        val1 = pl.DataFrame({"y": [11.0, 21.0], "y_hat": [10.0, 20.0]})  # MAE = 1
        val2 = pl.DataFrame({"y": [11.0, 21.0], "y_hat": [9.0, 19.0]})  # MAE = 2

        ens = WeightedEnsemble(weights="inverse_error")
        result = ens.combine([f1, f2], validation_dfs=[val1, val2])

        # w1 = (1/1) / (1/1 + 1/2) = 2/3, w2 = (1/2) / (1/1 + 1/2) = 1/3
        expected_0 = (2 / 3) * 10.0 + (1 / 3) * 20.0
        assert result["y_hat"][0] == pytest.approx(expected_0, abs=0.1)

    def test_multiple_series(self):
        f1 = pl.DataFrame(
            {
                "unique_id": ["A", "A", "B", "B"],
                "ds": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "y_hat": [10.0, 20.0, 100.0, 200.0],
            }
        )
        f2 = pl.DataFrame(
            {
                "unique_id": ["A", "A", "B", "B"],
                "ds": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "y_hat": [20.0, 40.0, 200.0, 400.0],
            }
        )
        ens = WeightedEnsemble(weights="equal")
        result = ens.combine([f1, f2])

        assert len(result) == 4
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a == pytest.approx([15.0, 30.0])

    def test_three_models(self):
        f1 = _make_forecast([10.0])
        f2 = _make_forecast([20.0])
        f3 = _make_forecast([30.0])
        ens = WeightedEnsemble(weights="equal")
        result = ens.combine([f1, f2, f3])

        assert result["y_hat"][0] == pytest.approx(20.0)

    def test_output_schema(self):
        f1 = _make_forecast([1.0])
        f2 = _make_forecast([2.0])
        result = WeightedEnsemble().combine([f1, f2])

        assert result.columns == ["unique_id", "ds", "y_hat"]

    def test_weight_length_mismatch(self):
        f1 = _make_forecast([1.0])
        f2 = _make_forecast([2.0])
        ens = WeightedEnsemble(weights=[0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="Expected 2 weights"):
            ens.combine([f1, f2])

    def test_mismatched_forecast_rows(self):
        f1 = _make_forecast([1.0, 2.0])
        f2 = _make_forecast([1.0])  # Different length
        ens = WeightedEnsemble()
        with pytest.raises(ValueError, match="different"):
            ens.combine([f1, f2])

    def test_inverse_error_no_validation(self):
        f1 = _make_forecast([1.0])
        f2 = _make_forecast([2.0])
        ens = WeightedEnsemble(weights="inverse_error")
        with pytest.raises(ValueError, match="validation_dfs"):
            ens.combine([f1, f2])

    def test_single_forecast_rejected(self):
        f1 = _make_forecast([1.0])
        ens = WeightedEnsemble()
        with pytest.raises(ValueError, match="at least 2"):
            ens.combine([f1])

    def test_empty_forecasts_rejected(self):
        ens = WeightedEnsemble()
        with pytest.raises(ValueError, match="non-empty"):
            ens.combine([])


def test_top_level_import():
    import polars_ts

    assert polars_ts.WeightedEnsemble is WeightedEnsemble


def test_submodule_import():
    from polars_ts.ensemble import WeightedEnsemble as WE

    assert callable(WE)
