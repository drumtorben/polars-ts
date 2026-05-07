"""Tests for KASBAClusterer class and kasba() convenience function (issue #196).

Covers the Polars-level API: DataFrame input/output, column types,
fit/predict lifecycle, univariate and multivariate modes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.kasba import KASBAClusterer, kasba


@pytest.fixture
def univariate_df() -> pl.DataFrame:
    """10 univariate series (2 clusters: ascending vs descending), 20 timepoints."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(5):
        vals = np.arange(20, dtype=np.float64) + rng.normal(0, 0.1, 20)
        for t, v in enumerate(vals):
            rows.append({"unique_id": f"up_{i}", "ds": t, "y": v})
    for i in range(5):
        vals = np.arange(19, -1, -1, dtype=np.float64) + rng.normal(0, 0.1, 20)
        for t, v in enumerate(vals):
            rows.append({"unique_id": f"down_{i}", "ds": t, "y": v})
    return pl.DataFrame(rows)


@pytest.fixture
def multivariate_df() -> pl.DataFrame:
    """6 multivariate series, 2 channels, 15 timepoints (2 clusters)."""
    rng = np.random.default_rng(123)
    rows = []
    for i in range(3):
        for ch in ["temp", "humidity"]:
            vals = rng.normal(0, 1, 15)
            for t, v in enumerate(vals):
                rows.append({"unique_id": f"a_{i}", "ds": t, "channel": ch, "y": v})
    for i in range(3):
        for ch in ["temp", "humidity"]:
            vals = rng.normal(10, 1, 15)
            for t, v in enumerate(vals):
                rows.append({"unique_id": f"b_{i}", "ds": t, "channel": ch, "y": v})
    return pl.DataFrame(rows)


class TestKASBAClustererFit:
    """Test KASBAClusterer.fit() returns correct structure."""

    def test_labels_is_dataframe(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert isinstance(clf.labels_, pl.DataFrame)

    def test_labels_columns(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.labels_.columns == ["unique_id", "cluster"]

    def test_labels_row_count(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.labels_.height == 10

    def test_cluster_values_in_range(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        clusters = clf.labels_["cluster"].to_list()
        assert all(0 <= c < 2 for c in clusters)

    def test_centroids_shape(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.centroids_ is not None
        assert clf.centroids_.shape == (2, 1 * 20)

    def test_inertia_non_negative(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.inertia_ >= 0.0

    def test_n_iter_positive(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.n_iter_ >= 1

    def test_is_fitted_flag(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        assert not clf._is_fitted
        clf.fit(univariate_df)
        assert clf._is_fitted

    def test_fit_returns_self(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        result = clf.fit(univariate_df)
        assert result is clf

    def test_deterministic_with_same_seed(self, univariate_df):
        clf1 = KASBAClusterer(n_clusters=2, seed=42)
        clf1.fit(univariate_df)
        clf2 = KASBAClusterer(n_clusters=2, seed=42)
        clf2.fit(univariate_df)
        assert clf1.labels_.equals(clf2.labels_)

    def test_id_dtype_preserved_string(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        assert clf.labels_["unique_id"].dtype == pl.String

    def test_id_dtype_preserved_int(self, univariate_df):
        df = univariate_df.with_columns(
            pl.col("unique_id").str.replace("up_", "1").str.replace("down_", "2").cast(pl.Int64).alias("unique_id")
        )
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(df)
        assert clf.labels_["unique_id"].dtype == pl.Int64


class TestKASBAClustererPredict:
    """Test KASBAClusterer.predict()."""

    def test_predict_before_fit_raises(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(univariate_df)

    def test_predict_returns_dataframe(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        preds = clf.predict(univariate_df)
        assert isinstance(preds, pl.DataFrame)
        assert preds.columns == ["unique_id", "cluster"]

    def test_predict_same_data_matches_fit(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        preds = clf.predict(univariate_df)
        assert clf.labels_.sort("unique_id").equals(preds.sort("unique_id"))

    def test_predict_new_data(self, univariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(univariate_df)
        # Create a single new ascending series
        new_df = pl.DataFrame(
            {
                "unique_id": ["new_up"] * 20,
                "ds": list(range(20)),
                "y": list(range(20)),
            }
        )
        preds = clf.predict(new_df)
        assert preds.height == 1
        assert 0 <= preds["cluster"][0] < 2


class TestKASBAClustererMultivariate:
    """Test multivariate mode via channel_col."""

    def test_multivariate_fit(self, multivariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(multivariate_df, channel_col="channel")
        assert clf.labels_.height == 6
        assert clf._n_channels == 2

    def test_multivariate_centroids_shape(self, multivariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(multivariate_df, channel_col="channel")
        assert clf.centroids_.shape == (2, 2 * 15)

    def test_multivariate_independent_vs_dependent(self, multivariate_df):
        clf_ind = KASBAClusterer(n_clusters=2, independent=True, seed=42)
        clf_ind.fit(multivariate_df, channel_col="channel")

        clf_dep = KASBAClusterer(n_clusters=2, independent=False, seed=42)
        clf_dep.fit(multivariate_df, channel_col="channel")

        # Both should produce valid results (may or may not differ)
        assert clf_ind.labels_.height == 6
        assert clf_dep.labels_.height == 6

    def test_multivariate_predict(self, multivariate_df):
        clf = KASBAClusterer(n_clusters=2)
        clf.fit(multivariate_df, channel_col="channel")
        preds = clf.predict(multivariate_df, channel_col="channel")
        assert preds.height == 6


class TestKasbaConvenienceFunction:
    """Test the kasba() top-level function."""

    def test_returns_dataframe(self, univariate_df):
        result = kasba(univariate_df, k=2)
        assert isinstance(result, pl.DataFrame)

    def test_columns(self, univariate_df):
        result = kasba(univariate_df, k=2)
        assert result.columns == ["unique_id", "cluster"]

    def test_row_count(self, univariate_df):
        result = kasba(univariate_df, k=2)
        assert result.height == 10

    def test_cluster_range(self, univariate_df):
        result = kasba(univariate_df, k=3)
        clusters = result["cluster"].to_list()
        assert all(0 <= c < 3 for c in clusters)

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "item": ["a"] * 10 + ["b"] * 10,
                "time": list(range(10)) * 2,
                "value": list(range(10)) + list(range(9, -1, -1)),
            }
        )
        result = kasba(df, k=2, id_col="item", target_col="value")
        assert result.columns == ["item", "cluster"]
        assert result.height == 2

    def test_multivariate(self, multivariate_df):
        result = kasba(multivariate_df, k=2, channel_col="channel")
        assert result.height == 6

    def test_deterministic(self, univariate_df):
        r1 = kasba(univariate_df, k=2, seed=42)
        r2 = kasba(univariate_df, k=2, seed=42)
        assert r1.sort("unique_id").equals(r2.sort("unique_id"))

    def test_kwargs_forwarded(self, univariate_df):
        result = kasba(univariate_df, k=2, c=2.0, ba_subset_size=0.3)
        assert result.height == 10


class TestKASBAImports:
    """Test that KASBA is properly registered in __init__.py."""

    def test_import_from_clustering(self):
        from polars_ts.clustering import KASBAClusterer as C1
        from polars_ts.clustering import kasba as f1

        assert C1 is KASBAClusterer
        # kasba may resolve to the module due to same-name module;
        # verify the callable is accessible
        assert callable(f1) or hasattr(f1, "kasba")

    def test_import_from_top_level(self):
        from polars_ts import KASBAClusterer as C2
        from polars_ts import kasba as f2

        assert C2 is KASBAClusterer
        assert f2 is kasba
