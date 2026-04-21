"""Tests for exponential smoothing forecasters (#49)."""

from datetime import date, timedelta

import polars as pl
import pytest

from polars_ts.models.exponential_smoothing import (
    holt_forecast,
    holt_winters_forecast,
    ses_forecast,
)


def _make_df(n: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [date(2024, 1, i + 1) for i in range(n)] * 2,
            "y": [float(i) + 10.0 for i in range(n)] + [float(2 * i) + 5.0 for i in range(n)],
        }
    )


class TestSES:
    def test_basic(self):
        result = ses_forecast(_make_df(), h=3, alpha=0.3)
        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_flat_forecast(self):
        result = ses_forecast(_make_df(n=20), h=3, alpha=0.3)
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[0] == pytest.approx(a[1])  # SES produces flat forecast
        assert a[1] == pytest.approx(a[2])

    def test_alpha_bounds(self):
        with pytest.raises(ValueError, match="alpha"):
            ses_forecast(_make_df(), h=1, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ses_forecast(_make_df(), h=1, alpha=1.0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="positive"):
            ses_forecast(_make_df(), h=0)


class TestHolt:
    def test_basic(self):
        result = holt_forecast(_make_df(), h=3)
        assert len(result) == 6

    def test_trend(self):
        result = holt_forecast(_make_df(), h=3, alpha=0.5, beta=0.3)
        a = result.filter(pl.col("unique_id") == "A")["y_hat"].to_list()
        assert a[1] > a[0]  # Upward trend

    def test_beta_bounds(self):
        with pytest.raises(ValueError, match="beta"):
            holt_forecast(_make_df(), h=1, beta=0.0)

    def test_short_series(self):
        df = pl.DataFrame({"unique_id": ["A"], "ds": [date(2024, 1, 1)], "y": [1.0]})
        with pytest.raises(ValueError, match="at least 2"):
            holt_forecast(df, h=1)


class TestHoltWinters:
    def _make_seasonal(self, n: int = 48, m: int = 12) -> pl.DataFrame:
        import numpy as np

        base = date(2024, 1, 1)
        values = [10.0 + 0.5 * i + 5.0 * np.sin(2 * np.pi * i / m) for i in range(n)]
        return pl.DataFrame(
            {
                "unique_id": ["A"] * n,
                "ds": [base + timedelta(days=i) for i in range(n)],
                "y": values,
            }
        )

    def test_additive(self):
        result = holt_winters_forecast(self._make_seasonal(), h=12, season_length=12, seasonal="additive")
        assert len(result) == 12

    def test_multiplicative(self):
        df = self._make_seasonal()
        # Shift to strictly positive for multiplicative
        df = df.with_columns((pl.col("y") + 20.0).alias("y"))
        result = holt_winters_forecast(df, h=6, season_length=12, seasonal="multiplicative")
        assert len(result) == 6

    def test_invalid_seasonal(self):
        with pytest.raises(ValueError, match="seasonal"):
            holt_winters_forecast(self._make_seasonal(), h=1, season_length=12, seasonal="invalid")

    def test_short_series(self):
        df = pl.DataFrame(
            {"unique_id": ["A"] * 10, "ds": [date(2024, 1, i + 1) for i in range(10)], "y": list(range(10))}
        )
        with pytest.raises(ValueError, match="2\\*season_length"):
            holt_winters_forecast(df, h=1, season_length=12)

    def test_gamma_bounds(self):
        with pytest.raises(ValueError, match="gamma"):
            holt_winters_forecast(self._make_seasonal(), h=1, season_length=12, gamma=0.0)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.ses_forecast is ses_forecast
    assert polars_ts.holt_forecast is holt_forecast
    assert polars_ts.holt_winters_forecast is holt_winters_forecast


class TestRustPythonEquivalence:
    """Verify Rust and Python ETS implementations produce identical results."""

    @pytest.fixture(autouse=True)
    def _require_rust(self):
        pytest.importorskip("polars_ts_rs")

    def _make_values(self, n: int = 50) -> list[float]:
        import numpy as np

        rng = np.random.default_rng(42)
        return (rng.normal(0, 1, n).cumsum() + 100).tolist()

    def _make_seasonal(self, n: int = 48, m: int = 12) -> list[float]:
        import numpy as np

        trend = np.linspace(10, 30, n)
        seasonal = 5.0 * np.sin(2 * np.pi * np.arange(n) / m)
        noise = np.random.default_rng(42).normal(0, 0.5, n)
        return (trend + seasonal + noise).tolist()

    def test_ses_equivalence(self):
        from polars_ts_rs import ets_ses

        from polars_ts.models.exponential_smoothing import _ses_python

        values = self._make_values()
        for alpha in [0.1, 0.3, 0.5, 0.9]:
            py = _ses_python(values, alpha, 5)
            rs = ets_ses(values, alpha, 5)
            assert py == pytest.approx(rs), f"SES mismatch at alpha={alpha}"

    def test_holt_equivalence(self):
        from polars_ts_rs import ets_holt

        from polars_ts.models.exponential_smoothing import _holt_python

        values = self._make_values()
        for alpha, beta in [(0.3, 0.1), (0.5, 0.3), (0.9, 0.5)]:
            py = _holt_python(values, alpha, beta, 5)
            rs = ets_holt(values, alpha, beta, 5)
            assert py == pytest.approx(rs), f"Holt mismatch at alpha={alpha}, beta={beta}"

    def test_hw_additive_equivalence(self):
        from polars_ts_rs import ets_holt_winters

        from polars_ts.models.exponential_smoothing import _hw_python

        values = self._make_seasonal(n=48, m=12)
        for alpha, beta, gamma in [(0.3, 0.1, 0.1), (0.5, 0.2, 0.3)]:
            py = _hw_python(values, alpha, beta, gamma, 12, True, 12)
            rs = ets_holt_winters(values, alpha, beta, gamma, 12, True, 12)
            assert py == pytest.approx(rs), f"HW additive mismatch at alpha={alpha}, beta={beta}, gamma={gamma}"

    def test_hw_multiplicative_equivalence(self):
        from polars_ts_rs import ets_holt_winters

        from polars_ts.models.exponential_smoothing import _hw_python

        # Shift positive for multiplicative
        values = [v + 50 for v in self._make_seasonal(n=48, m=12)]
        for alpha, beta, gamma in [(0.3, 0.1, 0.1), (0.5, 0.2, 0.3)]:
            py = _hw_python(values, alpha, beta, gamma, 12, False, 12)
            rs = ets_holt_winters(values, alpha, beta, gamma, 12, False, 12)
            assert py == pytest.approx(rs), f"HW multiplicative mismatch at alpha={alpha}, beta={beta}, gamma={gamma}"
