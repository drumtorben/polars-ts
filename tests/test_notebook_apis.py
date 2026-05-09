"""Test suite validating API contracts used across all 14 notebooks.

Each test validates the specific calling patterns from the notebooks,
ensuring they don't crash after the refactoring. Tests are offline-friendly
(no network I/O) and use minimal synthetic data.
"""

import datetime

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_df():
    """20 series of length 30 — mimics M4 hourly subset used in NB01-09."""
    rng = np.random.default_rng(42)
    rows = []
    base = datetime.datetime(2023, 1, 1)
    for i in range(20):
        sid = f"H{i+1}"
        for j in range(30):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + datetime.timedelta(hours=j),
                    "y": float(rng.normal(0, 1)),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def ts_df_no_time():
    """20 series of length 30 — integer ds for distance/clustering tests."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(20):
        sid = f"H{i+1}"
        for j in range(30):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": j,
                    "y": float(rng.normal(0, 1)),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def small_ts():
    """3 series for quick tests."""
    rng = np.random.default_rng(7)
    rows = []
    base = datetime.datetime(2023, 1, 1)
    for i, sid in enumerate(["A", "B", "C"]):
        for j in range(50):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + datetime.timedelta(hours=j),
                    "y": float(rng.normal(i * 10, 1)),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# NB 01: Data Wrangling — basic polars operations (no polars_ts API to test)
# ---------------------------------------------------------------------------


class TestNB01:
    def test_polars_import(self):
        import polars as pl

        assert hasattr(pl, "DataFrame")


# ---------------------------------------------------------------------------
# NB 02: Feature Engineering — transforms
# ---------------------------------------------------------------------------


class TestNB02:
    def test_lag_features(self, small_ts):
        # NB02 uses polars .shift() operations — just verify df shape
        result = small_ts.with_columns(pl.col("y").shift(1).over("unique_id").alias("y_lag1"))
        assert "y_lag1" in result.columns


# ---------------------------------------------------------------------------
# NB 03: Forecasting Fundamentals — SCUM import
# ---------------------------------------------------------------------------


class TestNB03:
    def test_scum_in_lazy_imports(self):
        """SCUM should be registered in _LAZY_IMPORTS even if statsforecast isn't installed."""
        from polars_ts import _LAZY_IMPORTS

        assert "SCUM" in _LAZY_IMPORTS


# ---------------------------------------------------------------------------
# NB 04: ML Forecasting Pipelines — mae with positional args
# ---------------------------------------------------------------------------


class TestNB04:
    def test_mae_positional_args(self):
        """NB04 calls mae(merged, 'y', 'y_hat') with positional args."""
        from polars_ts import mae

        merged = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_hat": [1.1, 2.2, 2.8, 4.1, 5.0],
            }
        )
        result = mae(merged, "y", "y_hat")
        assert isinstance(result, float)
        assert result > 0

    def test_rmse_positional_args(self):
        from polars_ts import rmse

        merged = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_hat": [1.1, 2.2, 2.8, 4.1, 5.0],
            }
        )
        result = rmse(merged, "y", "y_hat")
        assert isinstance(result, float)
        assert result > 0


# ---------------------------------------------------------------------------
# NB 05: Uncertainty & Calibration — no special API changes
# ---------------------------------------------------------------------------


class TestNB05:
    def test_mae_returns_float(self):
        from polars_ts import mae

        df = pl.DataFrame({"y": [1.0, 2.0], "y_hat": [1.5, 2.5]})
        assert isinstance(mae(df, "y", "y_hat"), float)


# ---------------------------------------------------------------------------
# NB 07: Similarity & Clustering — CRITICAL: DataFrame labels
# ---------------------------------------------------------------------------


class TestNB07:
    def test_silhouette_score_accepts_dataframe(self, ts_df_no_time):
        """silhouette_score must accept labels as pl.DataFrame, not list."""
        from polars_ts import kmedoids, silhouette_score

        clusters = kmedoids(ts_df_no_time, k=2, method="sbd")
        assert isinstance(clusters, pl.DataFrame)
        assert "unique_id" in clusters.columns
        assert "cluster" in clusters.columns

        score = silhouette_score(ts_df_no_time, clusters, method="sbd")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_davies_bouldin_accepts_dataframe(self, ts_df_no_time):
        from polars_ts import davies_bouldin_score, kmedoids

        clusters = kmedoids(ts_df_no_time, k=2, method="sbd")
        score = davies_bouldin_score(ts_df_no_time, clusters, method="sbd")
        assert isinstance(score, float)
        assert score >= 0

    def test_calinski_harabasz_accepts_dataframe(self, ts_df_no_time):
        from polars_ts import calinski_harabasz_score, kmedoids

        clusters = kmedoids(ts_df_no_time, k=2, method="sbd")
        score = calinski_harabasz_score(ts_df_no_time, clusters, method="sbd")
        assert isinstance(score, float)
        assert score > 0

    def test_silhouette_samples_accepts_dataframe(self, ts_df_no_time):
        from polars_ts import kmedoids, silhouette_samples

        clusters = kmedoids(ts_df_no_time, k=2, method="sbd")
        samples = silhouette_samples(ts_df_no_time, clusters, method="sbd")
        assert isinstance(samples, pl.DataFrame)

    def test_kshape_labels_is_dataframe(self, ts_df_no_time):
        """KShape.labels_ is already a DataFrame — pass directly to eval functions."""
        from polars_ts import KShape, silhouette_score

        kshape = KShape(n_clusters=2)
        kshape.fit(ts_df_no_time)

        # labels_ should be a DataFrame with unique_id and cluster columns
        assert isinstance(kshape.labels_, pl.DataFrame)
        assert "unique_id" in kshape.labels_.columns
        assert "cluster" in kshape.labels_.columns

        score = silhouette_score(ts_df_no_time, kshape.labels_, method="sbd")
        assert isinstance(score, float)

    def test_hdbscan_cluster(self, ts_df_no_time):
        from polars_ts import hdbscan_cluster

        clusters = hdbscan_cluster(ts_df_no_time, method="sbd", min_cluster_size=3)
        assert isinstance(clusters, pl.DataFrame)
        assert "cluster" in clusters.columns

    def test_dbscan_cluster(self, ts_df_no_time):
        from polars_ts import dbscan_cluster

        clusters = dbscan_cluster(ts_df_no_time, method="sbd", eps=5.0, min_samples=2)
        assert isinstance(clusters, pl.DataFrame)
        assert "cluster" in clusters.columns

    def test_spectral_cluster(self, ts_df_no_time):
        from polars_ts import spectral_cluster

        clusters = spectral_cluster(ts_df_no_time, k=2, method="sbd", sigma=1.0)
        assert isinstance(clusters, pl.DataFrame)
        assert "cluster" in clusters.columns

    def test_auto_cluster(self, ts_df_no_time):
        from polars_ts import auto_cluster

        result = auto_cluster(
            ts_df_no_time,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 4),
            metric="silhouette",
        )
        assert hasattr(result, "best_method")
        assert hasattr(result, "best_labels")
        assert isinstance(result.best_labels, pl.DataFrame)


# ---------------------------------------------------------------------------
# NB 09: Ensembles & Reconciliation — CRITICAL: mae join pattern + CV generator
# ---------------------------------------------------------------------------


class TestNB09:
    def test_mae_requires_single_dataframe(self):
        """mae(df, actual_col, predicted_col) — NOT mae(test, forecast)."""
        from polars_ts import mae

        merged = pl.DataFrame(
            {
                "unique_id": ["A"] * 3,
                "ds": [1, 2, 3],
                "y": [10.0, 20.0, 30.0],
                "y_hat": [11.0, 19.0, 31.0],
            }
        )
        err = mae(merged, "y", "y_hat")
        assert isinstance(err, float)
        assert abs(err - 1.0) < 1e-6

    def test_expanding_window_cv_is_generator(self, small_ts):
        """expanding_window_cv yields (train, test) tuples — NOT a function-call API."""
        from polars_ts import expanding_window_cv

        folds = list(expanding_window_cv(small_ts, n_splits=3, horizon=5))
        assert len(folds) == 3
        for tr, te in folds:
            assert isinstance(tr, pl.DataFrame)
            assert isinstance(te, pl.DataFrame)
            assert te.height > 0
            assert tr.height > 0

    def test_expanding_window_cv_no_forecast_fn_param(self):
        """expanding_window_cv does NOT accept forecast_fn kwarg."""
        import inspect

        from polars_ts import expanding_window_cv

        sig = inspect.signature(expanding_window_cv)
        params = list(sig.parameters.keys())
        assert "forecast_fn" not in params
        assert "n_splits" in params
        assert "horizon" in params

    def test_naive_forecast(self, small_ts):
        from polars_ts import naive_forecast

        fc = naive_forecast(small_ts, h=3)
        assert isinstance(fc, pl.DataFrame)
        assert "y_hat" in fc.columns

    def test_seasonal_naive_forecast(self, small_ts):
        from polars_ts import seasonal_naive_forecast

        fc = seasonal_naive_forecast(small_ts, h=3, season_length=7)
        assert isinstance(fc, pl.DataFrame)
        assert "y_hat" in fc.columns

    def test_weighted_ensemble(self, small_ts):
        from polars_ts import WeightedEnsemble, naive_forecast, seasonal_naive_forecast

        fc1 = naive_forecast(small_ts, h=3)
        fc2 = seasonal_naive_forecast(small_ts, h=3, season_length=7)

        ens = WeightedEnsemble(weights="equal")
        combined = ens.combine(forecasts=[fc1, fc2])
        assert isinstance(combined, pl.DataFrame)
        assert "y_hat" in combined.columns

    def test_reconcile(self):
        from polars_ts import naive_forecast, reconcile

        base = datetime.datetime(2023, 1, 1)
        n = 50
        hier_df = pl.DataFrame(
            {
                "unique_id": ["A"] * n + ["B"] * n + ["Total"] * n,
                "ds": [base + datetime.timedelta(hours=i) for i in range(n)] * 3,
                "y": list(range(n)) + list(range(n, 2 * n)) + list(range(0, 2 * n, 2))[:n],
            }
        )
        train = hier_df.filter(pl.col("ds") < base + datetime.timedelta(hours=40))
        fc = naive_forecast(train, h=10)

        hierarchy = {"Total": "A+B"}
        recon = reconcile(fc, hierarchy=hierarchy, method="bottom_up")
        assert isinstance(recon, pl.DataFrame)
        assert "y_hat" in recon.columns


# ---------------------------------------------------------------------------
# NB 11: Time Series Imaging — function names
# ---------------------------------------------------------------------------


class TestNB11:
    def test_to_recurrence_plot(self, ts_df_no_time):
        from polars_ts.imaging.recurrence import rqa_features, to_recurrence_plot

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_recurrence_plot(series, threshold=0.3)
        assert isinstance(images, dict)
        assert "H1" in images
        assert images["H1"].ndim == 2

        features = rqa_features(images["H1"])
        assert isinstance(features, dict)
        assert "recurrence_rate" in features

    def test_to_gasf(self, ts_df_no_time):
        from polars_ts.imaging.angular import to_gasf

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_gasf(series)
        assert isinstance(images, dict)
        assert "H1" in images
        assert images["H1"].ndim == 2

    def test_to_gadf(self, ts_df_no_time):
        from polars_ts.imaging.angular import to_gadf

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_gadf(series)
        assert isinstance(images, dict)
        assert "H1" in images

    def test_to_mtf(self, ts_df_no_time):
        from polars_ts.imaging.transition import to_mtf

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_mtf(series)
        assert isinstance(images, dict)
        assert "H1" in images

    def test_to_spectrogram(self, ts_df_no_time):
        from polars_ts.imaging.spectral import to_spectrogram

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_spectrogram(series)
        assert isinstance(images, dict)
        assert "H1" in images
        assert images["H1"].ndim == 2

    def test_to_scalogram(self, ts_df_no_time):
        from polars_ts.imaging.spectral import to_scalogram

        series = ts_df_no_time.filter(pl.col("unique_id") == "H1")
        images = to_scalogram(series)
        assert isinstance(images, dict)
        assert "H1" in images
        assert images["H1"].ndim == 2


# ---------------------------------------------------------------------------
# NB 12: Advanced Feature Extraction — signature_features, to_gasf, embeddings
# ---------------------------------------------------------------------------


class TestNB12:
    def test_signature_features(self, ts_df_no_time):
        from polars_ts.imaging.signature import signature_features

        sig_df = signature_features(ts_df_no_time, depth=3, augmentations=["time"])
        assert isinstance(sig_df, pl.DataFrame)
        assert "unique_id" in sig_df.columns
        assert sig_df.height == ts_df_no_time["unique_id"].n_unique()

    def test_to_gasf_returns_dict(self, ts_df_no_time):
        from polars_ts.imaging.angular import to_gasf

        images = to_gasf(ts_df_no_time)
        assert isinstance(images, dict)
        assert len(images) == ts_df_no_time["unique_id"].n_unique()

    def test_rocket_features(self, ts_df_no_time):
        from polars_ts import rocket_features

        feat = rocket_features(ts_df_no_time, n_kernels=10, seed=42)
        assert isinstance(feat, pl.DataFrame)
        assert "unique_id" in feat.columns
        # n_kernels * 2 features + unique_id
        assert feat.width > 2

    def test_shapelet_cluster(self, ts_df_no_time):
        from polars_ts import shapelet_cluster

        labels = shapelet_cluster(ts_df_no_time, k=2, n_shapelets=3, seed=42)
        assert isinstance(labels, pl.DataFrame)
        assert "cluster" in labels.columns


# ---------------------------------------------------------------------------
# NB 13: Agentic Forecasting — agent imports
# ---------------------------------------------------------------------------


class TestNB13:
    def test_agents_import(self):
        from polars_ts.agents import (
            CuratorAgent,
            ForecasterAgent,
            PlannerAgent,
            TimeSeriesScientist,
        )

        assert callable(TimeSeriesScientist)
        assert callable(CuratorAgent)
        assert callable(PlannerAgent)
        assert callable(ForecasterAgent)

    def test_curator_agent(self, small_ts):
        from polars_ts.agents import CuratorAgent

        curator = CuratorAgent()
        report = curator.curate(small_ts)
        assert hasattr(report, "n_observations")
        assert hasattr(report, "n_missing")

    def test_time_series_scientist(self, small_ts):
        from polars_ts.agents import TimeSeriesScientist

        scientist = TimeSeriesScientist(horizon=3)
        result = scientist.run(small_ts)
        assert hasattr(result, "predictions")
        assert hasattr(result, "report")
        assert isinstance(result.predictions, pl.DataFrame)


# ---------------------------------------------------------------------------
# NB 14: KASBA Clustering
# ---------------------------------------------------------------------------


class TestNB14:
    def test_kasba_clusterer(self, ts_df_no_time):
        from polars_ts.clustering.kasba import KASBAClusterer

        kasba = KASBAClusterer(n_clusters=2)
        kasba.fit(ts_df_no_time)
        assert hasattr(kasba, "centroids_")
        assert hasattr(kasba, "labels_")
        # NB14 indexes centroids as ndarray
        assert kasba.centroids_[0] is not None
