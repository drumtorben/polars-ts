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


def test_multiple_series():
    """garch_fit returns one result per series."""
    df1 = _make_garch_data(n=300, seed=42)
    df2 = _make_garch_data(n=300, seed=99).with_columns(pl.lit("B").alias("unique_id"))
    df = pl.concat([df1, df2])
    results = garch_fit(df)
    assert "A" in results
    assert "B" in results
    assert isinstance(results["A"], GARCHResult)
    assert isinstance(results["B"], GARCHResult)


def test_p2_q1():
    """GARCH(2,1) should fit without error."""
    df = _make_garch_data(n=400, seed=7)
    results = garch_fit(df, p=2, q=1)
    r = results["A"]
    assert r.p == 2
    assert r.q == 1
    assert len(r.alpha) == 2
    assert len(r.beta) == 1


def test_p1_q2():
    """GARCH(1,2) should fit without error."""
    df = _make_garch_data(n=400, seed=8)
    results = garch_fit(df, p=1, q=2)
    r = results["A"]
    assert r.p == 1
    assert r.q == 2
    assert len(r.alpha) == 1
    assert len(r.beta) == 2


def test_too_short_series():
    """Series shorter than p+q+1 should raise ValueError."""
    df = pl.DataFrame({"unique_id": ["A"] * 2, "ds": [0, 1], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="too short"):
        garch_fit(df, p=1, q=1)


def test_constant_series():
    """Constant series (zero variance) should still fit without crashing."""
    df = pl.DataFrame({"unique_id": ["A"] * 50, "ds": list(range(50)), "y": [1.0] * 50})
    results = garch_fit(df)
    r = results["A"]
    # omega should be near zero for constant data
    assert r.omega >= 0


def test_custom_column_names():
    """garch_fit should work with non-default column names."""
    df = _make_garch_data().rename({"unique_id": "series", "ds": "time", "y": "value"})
    results = garch_fit(df, target_col="value", id_col="series", time_col="time")
    assert "A" in results


def test_long_horizon_convergence():
    """Forecast variance should converge toward unconditional variance."""
    df = _make_garch_data(n=500, seed=42)
    results = garch_fit(df)
    r = results["A"]
    fc = garch_forecast(r, horizon=200)
    # Last values should be very close to each other (convergence)
    assert abs(fc[-1] - fc[-2]) < 1e-6


def test_mean_reversion():
    """Variance forecast should revert toward long-run mean."""
    df = _make_garch_data(n=500, seed=42)
    results = garch_fit(df)
    r = results["A"]
    # Long-run (unconditional) variance: omega / (1 - sum(alpha) - sum(beta))
    persistence = sum(r.alpha) + sum(r.beta)
    if persistence < 1:
        unconditional = r.omega / (1 - persistence)
        fc = garch_forecast(r, horizon=100)
        # Last forecast should be close to unconditional
        assert fc[-1] == pytest.approx(unconditional, rel=0.3)


def test_invalid_q():
    """Negative q should raise ValueError."""
    with pytest.raises(ValueError, match="q must"):
        garch_fit(_make_garch_data(), q=-1)


def test_forecast_all_positive():
    """All forecast values must be strictly positive."""
    df = _make_garch_data()
    results = garch_fit(df)
    fc = garch_forecast(results["A"], horizon=20)
    assert all(v > 0 for v in fc)


def test_top_level_imports():
    import polars_ts

    assert polars_ts.garch_fit is garch_fit
    assert polars_ts.garch_forecast is garch_forecast
