"""Tests for GARCH volatility modelling (#51)."""

import numpy as np
import polars as pl
import pytest

scipy = pytest.importorskip("scipy")

from polars_ts.volatility import GARCHResult, garch_fit, garch_forecast  # noqa: E402


def _make_garch_data(n: int = 300, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.01, 0.1, 0.85
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return pl.DataFrame({"unique_id": ["A"] * n, "ds": list(range(n)), "y": eps.tolist()})


class TestGARCH:
    def test_fit_basic(self):
        df = _make_garch_data()
        results = garch_fit(df)
        assert "A" in results
        r = results["A"]
        assert isinstance(r, GARCHResult)
        assert r.p == 1
        assert r.q == 1

    def test_parameters_reasonable(self):
        df = _make_garch_data(n=500)
        results = garch_fit(df)
        r = results["A"]
        # alpha + beta should be close to 0.95 (0.1 + 0.85)
        assert 0.5 < sum(r.alpha) + sum(r.beta) < 1.0

    def test_conditional_variance_stored(self):
        df = _make_garch_data()
        results = garch_fit(df)
        assert len(results["A"].conditional_variance) == 300

    def test_forecast(self):
        df = _make_garch_data()
        results = garch_fit(df)
        fc = garch_forecast(results["A"], horizon=5)
        assert len(fc) == 5
        assert all(v > 0 for v in fc)

    def test_forecast_converges(self):
        df = _make_garch_data()
        results = garch_fit(df)
        fc = garch_forecast(results["A"], horizon=50)
        # Long-run variance should converge
        assert abs(fc[-1] - fc[-2]) < abs(fc[1] - fc[0]) + 1e-10

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p must"):
            garch_fit(_make_garch_data(), p=0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="positive"):
            garch_forecast(GARCHResult(0.01, [0.1], [0.8], 1, 1, [1.0] * 10), horizon=0)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.garch_fit is garch_fit
    assert polars_ts.garch_forecast is garch_forecast
