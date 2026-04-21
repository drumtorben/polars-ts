"""Tests for SCUM ensemble model (polars_ts/models/scum.py). Issue #78."""

import numpy as np
import polars as pl
import pytest

forecast_available = True
try:
    import statsforecast  # noqa: F401
    import utilsforecast  # noqa: F401

    from polars_ts.models.scum import SCUM
except ImportError:
    forecast_available = False
    SCUM = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    not forecast_available, reason="statsforecast/utilsforecast not installed"
)


@pytest.fixture
def sample_df():
    """Multi-series hourly DataFrame with clear seasonality."""
    np.random.seed(42)
    n = 100
    t = np.arange(n, dtype=float)
    rows = []
    for uid in ["H1", "H2", "H3"]:
        seasonal = 10 * np.sin(2 * np.pi * t / 24)
        trend = 0.05 * t
        noise = np.random.normal(0, 1, n)
        y = 50 + seasonal + trend + noise
        for i in range(n):
            rows.append({"unique_id": uid, "ds": i, "y": float(y[i])})
    return pl.DataFrame(rows)


class TestSCUM:
    def test_fit_returns_self(self):
        model = SCUM(season_length=24)
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 50
        result = model.fit(y)
        assert result is model

    def test_sub_models_populated_after_fit(self):
        model = SCUM(season_length=24)
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 50
        model.fit(y)
        assert model.sub_models is not None
        assert len(model.sub_models) == 4  # AutoARIMA, AutoETS, AutoCES, DynamicOptimizedTheta

    def test_predict_without_fit_raises(self):
        model = SCUM(season_length=24)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(h=5)

    def test_predict_returns_mean(self):
        model = SCUM(season_length=24)
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 50
        model.fit(y)
        result = model.predict(h=5)
        assert "mean" in result
        assert len(result["mean"]) == 5

    def test_predict_with_level(self):
        model = SCUM(season_length=24)
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) + 50
        model.fit(y)
        result = model.predict(h=5, level=(95,))
        assert "mean" in result
        assert "lo-95" in result
        assert "hi-95" in result
        assert len(result["mean"]) == 5

    def test_custom_alias(self):
        model = SCUM(season_length=12, alias="MyEnsemble")
        assert model.alias == "MyEnsemble"


class TestSCUMWithStatsForecast:
    def test_statsforecast_fit_predict(self, sample_df):
        from statsforecast import StatsForecast

        sf = StatsForecast(
            models=[SCUM(season_length=24)],
            freq=1,
            n_jobs=1,
        )
        sf.fit(df=sample_df)
        fc = sf.predict(h=5)
        assert "SCUM" in fc.columns
        assert fc["unique_id"].n_unique() == 3
        assert len(fc) == 15  # 3 series * 5 steps

    def test_multi_model_comparison(self, sample_df):
        """SCUM alongside another model in StatsForecast."""
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive

        sf = StatsForecast(
            models=[SCUM(season_length=24), SeasonalNaive(season_length=24)],
            freq=1,
            n_jobs=1,
        )
        sf.fit(df=sample_df)
        fc = sf.predict(h=5)
        assert "SCUM" in fc.columns
        assert "SeasonalNaive" in fc.columns

    def test_prediction_intervals(self, sample_df):
        """SCUM should produce prediction intervals when level is specified."""
        from statsforecast import StatsForecast

        sf = StatsForecast(
            models=[SCUM(season_length=24)],
            freq=1,
            n_jobs=1,
        )
        sf.fit(df=sample_df)
        fc = sf.predict(h=5, level=[95])
        assert "SCUM-lo-95" in fc.columns
        assert "SCUM-hi-95" in fc.columns
        # Lower bound should be below upper bound
        assert (fc["SCUM-lo-95"] <= fc["SCUM-hi-95"]).all()


def test_top_level_import():
    import polars_ts

    assert polars_ts.SCUM is SCUM
