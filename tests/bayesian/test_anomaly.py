"""Tests for Bayesian anomaly scoring (#121)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")

from polars_ts.bayesian.anomaly import (  # noqa: E402
    BayesianAnomalyDetector,
    BayesianAnomalyResult,
    _compute_bayes_factor,
    _compute_pvalue,
    _NIGState,
    bayesian_anomaly_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = date(2024, 1, 1)
    values = 10.0 + rng.normal(0, 0.5, n)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


def _make_df_with_anomaly(n: int = 50, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = date(2024, 1, 1)
    values = 10.0 + rng.normal(0, 0.5, n)
    # Inject anomalies at positions 30 and 35
    values[30] = 50.0
    values[35] = -30.0
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
        }
    )


def _make_multi_df(n: int = 30, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = date(2024, 1, 1)
    vals_a = 10.0 + rng.normal(0, 0.5, n)
    vals_b = 20.0 + rng.normal(0, 0.3, n)
    vals_b[20] = 100.0  # anomaly in B
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n + ["B"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)] * 2,
            "y": vals_a.tolist() + vals_b.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# NIG state tests
# ---------------------------------------------------------------------------


class TestNIGState:
    def test_initial_values(self):
        s = _NIGState(mu=10.0, kappa=1.0, alpha=2.0, beta=1.0)
        assert s.mu == 10.0
        assert s.kappa == 1.0

    def test_update_increases_kappa(self):
        s = _NIGState()
        s.update(5.0)
        assert s.kappa == 2.0

    def test_update_shifts_mean(self):
        s = _NIGState(mu=0.0, kappa=1.0)
        s.update(10.0)
        assert s.mu == pytest.approx(5.0)

    def test_predictive_params_finite(self):
        s = _NIGState(mu=10.0, kappa=5.0, alpha=5.0, beta=2.0)
        mean, scale = s.predictive_params()
        assert np.isfinite(mean)
        assert scale > 0

    def test_predictive_df(self):
        s = _NIGState(alpha=3.0)
        assert s.predictive_df() == 6.0


# ---------------------------------------------------------------------------
# P-value and Bayes factor tests
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    def test_pvalue_normal_observation(self):
        s = _NIGState(mu=10.0, kappa=100.0, alpha=50.0, beta=25.0)
        p = _compute_pvalue(10.0, s)
        assert p > 0.5  # perfectly expected value

    def test_pvalue_extreme_observation(self):
        s = _NIGState(mu=10.0, kappa=100.0, alpha=50.0, beta=25.0)
        p = _compute_pvalue(100.0, s)
        assert p < 0.01

    def test_pvalue_range(self):
        s = _NIGState(mu=10.0, kappa=10.0, alpha=5.0, beta=2.0)
        p = _compute_pvalue(10.5, s)
        assert 0 <= p <= 1

    def test_bayes_factor_normal(self):
        s = _NIGState(mu=10.0, kappa=100.0, alpha=50.0, beta=25.0)
        bf = _compute_bayes_factor(10.0, s)
        assert bf > 1  # favors normal model

    def test_bayes_factor_anomaly(self):
        s = _NIGState(mu=10.0, kappa=100.0, alpha=50.0, beta=25.0)
        bf = _compute_bayes_factor(100.0, s)
        assert bf < 1  # favors anomaly model


# ---------------------------------------------------------------------------
# BayesianAnomalyDetector validation
# ---------------------------------------------------------------------------


class TestDetectorValidation:
    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            BayesianAnomalyDetector(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            BayesianAnomalyDetector(threshold=1.0)

    def test_invalid_warmup(self):
        with pytest.raises(ValueError, match="warmup"):
            BayesianAnomalyDetector(warmup=-1)

    def test_invalid_anomaly_scale(self):
        with pytest.raises(ValueError, match="anomaly_scale"):
            BayesianAnomalyDetector(anomaly_scale=0.5)


# ---------------------------------------------------------------------------
# BayesianAnomalyDetector scoring
# ---------------------------------------------------------------------------


class TestDetectorScoring:
    def test_output_columns(self):
        det = BayesianAnomalyDetector()
        result = det.score(_make_df())
        expected = ["unique_id", "t", "value", "p_value", "bayes_factor", "is_anomaly"]
        assert result.scores.columns == expected

    def test_output_length(self):
        df = _make_df(n=30)
        det = BayesianAnomalyDetector()
        result = det.score(df)
        assert len(result.scores) == 30

    def test_pvalues_in_range(self):
        det = BayesianAnomalyDetector()
        result = det.score(_make_df())
        p_values = result.scores["p_value"].to_numpy()
        assert np.all(p_values >= 0)
        assert np.all(p_values <= 1)

    def test_warmup_no_anomalies(self):
        det = BayesianAnomalyDetector(warmup=10)
        result = det.score(_make_df())
        warmup_rows = result.scores.filter(pl.col("t") < 10)
        assert not warmup_rows["is_anomaly"].any()

    def test_detects_anomalies(self):
        det = BayesianAnomalyDetector(threshold=0.01, warmup=10)
        result = det.score(_make_df_with_anomaly())
        assert result.n_anomalies > 0
        anomalous = result.scores.filter(pl.col("is_anomaly"))
        # Should detect the injected anomalies (at t=30 and/or t=35)
        anomaly_times = anomalous["t"].to_list()
        assert 30 in anomaly_times or 35 in anomaly_times

    def test_no_anomalies_in_clean_data(self):
        det = BayesianAnomalyDetector(threshold=0.001, warmup=10)
        result = det.score(_make_df())
        # Clean data should have very few (if any) anomalies
        assert result.n_anomalies <= 3

    def test_multi_group(self):
        det = BayesianAnomalyDetector(warmup=5)
        result = det.score(_make_multi_df())
        assert len(result.scores) == 60  # 30 per group
        groups = result.scores["unique_id"].unique().to_list()
        assert sorted(groups) == ["A", "B"]

    def test_bayes_factors_positive(self):
        det = BayesianAnomalyDetector()
        result = det.score(_make_df())
        bf = result.scores["bayes_factor"].to_numpy()
        assert np.all(bf > 0)

    def test_lower_threshold_fewer_anomalies(self):
        df = _make_df_with_anomaly()
        r_loose = BayesianAnomalyDetector(threshold=0.1, warmup=10).score(df)
        r_strict = BayesianAnomalyDetector(threshold=0.001, warmup=10).score(df)
        assert r_strict.n_anomalies <= r_loose.n_anomalies

    def test_custom_prior(self):
        det = BayesianAnomalyDetector(prior_mu=10.0, prior_kappa=5.0, prior_alpha=5.0, prior_beta=2.0)
        result = det.score(_make_df())
        assert len(result.scores) == 50


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    def test_basic(self):
        result = bayesian_anomaly_score(_make_df())
        assert "p_value" in result.columns
        assert "bayes_factor" in result.columns
        assert "is_anomaly" in result.columns

    def test_with_anomalies(self):
        result = bayesian_anomaly_score(_make_df_with_anomaly(), threshold=0.01)
        assert result["is_anomaly"].any()

    def test_multi_group(self):
        result = bayesian_anomaly_score(_make_multi_df())
        assert len(result) == 60

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "sid": ["X"] * 20,
                "ts": [date(2024, 1, 1) + timedelta(days=i) for i in range(20)],
                "val": [float(i) for i in range(20)],
            }
        )
        result = bayesian_anomaly_score(df, id_col="sid", target_col="val", time_col="ts")
        assert "sid" in result.columns
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestBayesianAnomalyResult:
    def test_fields(self):
        r = BayesianAnomalyResult(scores=pl.DataFrame(), n_anomalies=5)
        assert r.n_anomalies == 5


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


def test_submodule_imports():
    from polars_ts.bayesian.anomaly import BayesianAnomalyDetector as BAD
    from polars_ts.bayesian.anomaly import bayesian_anomaly_score as bas

    assert BAD is BayesianAnomalyDetector
    assert bas is bayesian_anomaly_score
