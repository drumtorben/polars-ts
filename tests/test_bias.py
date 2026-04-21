"""Tests for forecast bias correction (#56)."""

import polars as pl
import pytest

from polars_ts.bias import bias_correct, bias_detect


def _make_biased_df(bias: float = 2.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "y": [float(i) for i in range(10)],
            "y_hat": [float(i) + bias for i in range(10)],
        }
    )


class TestBiasDetect:
    def test_positive_bias(self):
        result = bias_detect(_make_biased_df(2.0))
        assert result["mean_error"][0] == pytest.approx(2.0)
        assert result["sign_ratio"][0] == pytest.approx(1.0)  # All over-predictions

    def test_no_bias(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "y_hat": [1.0, 2.0, 3.0]})
        result = bias_detect(df)
        assert result["mean_error"][0] == pytest.approx(0.0)

    def test_per_group(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A", "B", "B"],
                "y": [1.0, 2.0, 1.0, 2.0],
                "y_hat": [3.0, 4.0, 0.5, 1.5],
            }
        )
        result = bias_detect(df, id_col="unique_id")
        assert len(result) == 2


class TestBiasCorrect:
    def test_mean_correction(self):
        result = bias_correct(_make_biased_df(2.0), method="mean")
        assert "y_hat_original" in result.columns
        # After correction, mean error should be ~0
        me = (result["y_hat"] - result["y"]).mean()
        assert me == pytest.approx(0.0, abs=0.1)

    def test_regression_correction(self):
        result = bias_correct(_make_biased_df(2.0), method="regression")
        me = (result["y_hat"] - result["y"]).mean()
        assert abs(me) < 0.5

    def test_quantile_correction(self):
        result = bias_correct(_make_biased_df(2.0), method="quantile")
        assert result["y_hat"].null_count() == 0

    def test_per_group(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A", "A", "A", "B", "B", "B"],
                "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "y_hat": [3.0, 4.0, 5.0, 8.0, 18.0, 28.0],
            }
        )
        result = bias_correct(df, method="mean", id_col="unique_id")
        assert len(result) == 6

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            bias_correct(_make_biased_df(), method="invalid")


def test_bias_negative():
    """Negative bias (under-predictions) should show negative mean_error."""
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 5,
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y_hat": [8.0, 18.0, 28.0, 38.0, 48.0],
        }
    )
    result = bias_detect(df)
    assert result["mean_error"][0] == pytest.approx(-2.0)


def test_bias_correct_preserves_length():
    """Correction should not change DataFrame length."""
    df = _make_biased_df(3.0)
    result = bias_correct(df, method="mean")
    assert len(result) == len(df)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.bias_detect is bias_detect
    assert polars_ts.bias_correct is bias_correct
