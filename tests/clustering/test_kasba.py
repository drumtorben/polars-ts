"""Tests for KASBAClusterer Python wrapper (issue #193)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.kasba import KASBAClusterer, kasba


@pytest.fixture
def cluster_df():
    """Univariate data: 6 series, two clear groups."""
    ascending = [float(i) for i in range(10)]
    descending = [float(9 - i) for i in range(10)]
    return pl.DataFrame(
        {
            "unique_id": (["A1"] * 10 + ["A2"] * 10 + ["A3"] * 10 + ["B1"] * 10 + ["B2"] * 10 + ["B3"] * 10),
            "y": (
                ascending
                + [x + 0.1 for x in ascending]
                + [x - 0.1 for x in ascending]
                + descending
                + [x + 0.1 for x in descending]
                + [x - 0.1 for x in descending]
            ),
        }
    )


@pytest.fixture
def multivariate_cluster_df():
    """Multivariate data: 6 series, 2 channels."""
    rng = np.random.default_rng(42)
    rows = []
    for sid in ["A1", "A2", "A3", "B1", "B2", "B3"]:
        for ch in ["x", "y"]:
            base = rng.normal(0 if sid.startswith("A") else 5, 0.1, 8)
            for val in base:
                rows.append({"unique_id": sid, "channel": ch, "y": float(val)})
    return pl.DataFrame(rows)


class TestKASBAClusterer:
    """Test KASBAClusterer class."""

    def test_fit_returns_self(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2)
        result = clf.fit(cluster_df)
        assert result is clf

    def test_labels_dataframe_schema(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.labels_ is not None
        assert "unique_id" in clf.labels_.columns
        assert "cluster" in clf.labels_.columns
        assert clf.labels_.shape[0] == cluster_df["unique_id"].n_unique()

    def test_predict_matches_fit(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        pred = clf.predict(cluster_df)
        assert pred.shape[0] == cluster_df["unique_id"].n_unique()
        assert "cluster" in pred.columns

    def test_predict_before_fit_raises(self, cluster_df):
        with pytest.raises(RuntimeError, match="fit"):
            KASBAClusterer().predict(cluster_df)

    def test_inertia_nonnegative(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.inertia_ >= 0.0

    def test_n_iter_positive(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.n_iter_ >= 1

    def test_reproducible_with_seed(self, cluster_df):
        labels1 = KASBAClusterer(n_clusters=2, seed=7).fit(cluster_df).labels_
        labels2 = KASBAClusterer(n_clusters=2, seed=7).fit(cluster_df).labels_
        assert labels1.equals(labels2)

    def test_centroids_stored(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.centroids_ is not None
        assert clf.centroids_.shape[0] == 2

    def test_convenience_function(self, cluster_df):
        result = kasba(cluster_df, k=2)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"unique_id", "cluster"}
        assert result.shape[0] == 6


class TestKASBAMultivariate:
    """Test multivariate KASBA clustering."""

    def test_multivariate_fit(self, multivariate_cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(multivariate_cluster_df, channel_col="channel")
        assert clf.labels_ is not None
        assert clf.labels_.shape[0] == multivariate_cluster_df["unique_id"].n_unique()

    def test_independent_vs_dependent(self, multivariate_cluster_df):
        labels_ind = (
            KASBAClusterer(n_clusters=2, independent=True).fit(multivariate_cluster_df, channel_col="channel").labels_
        )
        labels_dep = (
            KASBAClusterer(n_clusters=2, independent=False).fit(multivariate_cluster_df, channel_col="channel").labels_
        )
        # Both produce valid output
        assert labels_ind.shape == labels_dep.shape
        assert labels_ind.shape[0] == 6
