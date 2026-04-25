import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.dba import dba
from polars_ts.clustering.kmeans import TimeSeriesKMeans, kmeans_dba

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_data():
    """Four series in two well-separated groups (ascending vs descending)."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6,
            "y": (
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # ascending A
                + [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]  # ascending B
                + [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # descending C
                + [6.1, 5.1, 4.1, 3.1, 2.1, 1.1]  # descending D
            ),
        }
    )


@pytest.fixture
def three_cluster_data():
    """Six series in three groups."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8 + ["E"] * 8 + ["F"] * 8,
            "y": (
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # ascending A
                + [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]  # ascending B
                + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # descending C
                + [8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1]  # descending D
                + [1.0, 8.0, 1.0, 8.0, 1.0, 8.0, 1.0, 8.0]  # zigzag E
                + [1.1, 7.9, 1.1, 7.9, 1.1, 7.9, 1.1, 7.9]  # zigzag F
            ),
        }
    )


# ---------------------------------------------------------------------------
# DBA unit tests
# ---------------------------------------------------------------------------


class TestDBA:
    def test_identical_series_returns_same(self):
        """DBA of identical series should return the same series."""
        s = np.array([1.0, 2.0, 3.0, 4.0])
        result = dba([s, s.copy(), s.copy()])
        np.testing.assert_allclose(result, s, atol=1e-10)

    def test_two_series_average(self):
        """DBA of two aligned series produces a reasonable centroid."""
        s1 = np.array([0.0, 2.0, 4.0])
        s2 = np.array([2.0, 4.0, 6.0])
        result = dba([s1, s2])
        assert result.shape == (3,)
        # Centroid should be between the two series (element-wise)
        assert np.all(result >= s1.min())
        assert np.all(result <= s2.max())

    def test_single_series(self):
        """DBA of a single series returns a copy."""
        s = np.array([1.0, 2.0, 3.0])
        result = dba([s])
        np.testing.assert_array_equal(result, s)
        assert result is not s  # must be a copy

    def test_empty_list(self):
        """DBA of empty list returns empty array."""
        result = dba([])
        assert len(result) == 0

    def test_convergence(self):
        """DBA should converge within max_iter."""
        rng = np.random.default_rng(0)
        series = [rng.standard_normal(10) for _ in range(5)]
        result = dba(series, max_iter=100)
        assert result.shape == (10,)

    def test_unequal_length_series(self):
        """DBA handles series of different lengths."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dba([s1, s2])
        # Result length matches the medoid init (longer series)
        assert len(result) > 0

    def test_custom_init(self):
        """DBA with a custom initial centroid."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([3.0, 4.0, 5.0])
        init = np.array([2.0, 3.0, 4.0])
        result = dba([s1, s2], init=init)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0], atol=1e-10)

    def test_alignment_path_no_negative_indices(self):
        """DTW alignment path should never produce negative indices."""
        from polars_ts.clustering.dba import _dtw_alignment_path

        s = np.array([1.0, 2.0])
        t = np.array([1.0, 2.0, 3.0, 4.0])
        path = _dtw_alignment_path(s, t)
        for ci, si in path:
            assert ci >= 0, f"Negative centroid index: {ci}"
            assert si >= 0, f"Negative series index: {si}"
        # Path must start at (0,0) and end at (n-1, m-1)
        assert path[0] == (0, 0)
        assert path[-1] == (len(s) - 1, len(t) - 1)


# ---------------------------------------------------------------------------
# TimeSeriesKMeans class tests
# ---------------------------------------------------------------------------


class TestTimeSeriesKMeans:
    def test_fit_returns_self(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, max_iter=10)
        result = km.fit(cluster_data)
        assert result is km

    def test_labels_shape(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, max_iter=10)
        km.fit(cluster_data)
        assert km.labels_ is not None
        assert km.labels_.shape[0] == 4
        assert "unique_id" in km.labels_.columns
        assert "cluster" in km.labels_.columns

    def test_centroids_count(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, max_iter=10)
        km.fit(cluster_data)
        assert len(km.centroids_) == 2

    def test_similar_series_grouped(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, max_iter=50, seed=42)
        km.fit(cluster_data)
        labels = dict(zip(km.labels_["unique_id"].to_list(), km.labels_["cluster"].to_list(), strict=False))
        # A and B are ascending, C and D are descending
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_single_cluster(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=1, max_iter=10)
        km.fit(cluster_data)
        labels = km.labels_["cluster"].to_list()
        assert all(label == 0 for label in labels)

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        km = TimeSeriesKMeans(n_clusters=5)
        with pytest.raises(ValueError, match="Cannot create"):
            km.fit(df)

    def test_seed_reproducibility(self, cluster_data):
        km1 = TimeSeriesKMeans(n_clusters=2, max_iter=20, seed=123)
        km1.fit(cluster_data)
        km2 = TimeSeriesKMeans(n_clusters=2, max_iter=20, seed=123)
        km2.fit(cluster_data)
        assert km1.labels_["cluster"].to_list() == km2.labels_["cluster"].to_list()

    def test_centroids_are_valid_arrays(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, max_iter=10)
        km.fit(cluster_data)
        for c in km.centroids_:
            assert isinstance(c, np.ndarray)
            assert len(c) == 6  # same length as input series
            assert np.all(np.isfinite(c))

    def test_three_clusters(self, three_cluster_data):
        km = TimeSeriesKMeans(n_clusters=3, max_iter=50, seed=42)
        km.fit(three_cluster_data)
        assert km.labels_ is not None
        assert km.labels_["cluster"].n_unique() == 3
        labels = dict(zip(km.labels_["unique_id"].to_list(), km.labels_["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["E"] == labels["F"]

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "series_id": ["X"] * 4 + ["Y"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        km = TimeSeriesKMeans(n_clusters=2, max_iter=10)
        km.fit(df, id_col="series_id", target_col="value")
        assert km.labels_ is not None
        assert "series_id" in km.labels_.columns

    def test_unsupported_metric_raises(self, cluster_data):
        km = TimeSeriesKMeans(n_clusters=2, metric="erp")
        with pytest.raises(ValueError, match="Only metric='dtw'"):
            km.fit(cluster_data)

    def test_identical_series_same_cluster(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0] * 2 + [4.0, 3.0, 2.0, 1.0],
            }
        )
        km = TimeSeriesKMeans(n_clusters=2, max_iter=20)
        km.fit(df)
        labels = dict(zip(km.labels_["unique_id"].to_list(), km.labels_["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]


# ---------------------------------------------------------------------------
# kmeans_dba convenience function tests
# ---------------------------------------------------------------------------


class TestKmeansDbaFunction:
    def test_returns_dataframe(self, cluster_data):
        result = kmeans_dba(cluster_data, k=2, max_iter=10)
        assert isinstance(result, pl.DataFrame)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 4

    def test_correct_clustering(self, cluster_data):
        result = kmeans_dba(cluster_data, k=2, max_iter=50, seed=42)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["X"] * 4 + ["Y"] * 4,
                "val": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = kmeans_dba(df, k=2, max_iter=10, id_col="ts_id", target_col="val")
        assert "ts_id" in result.columns


# ---------------------------------------------------------------------------
# Top-level import tests
# ---------------------------------------------------------------------------


def test_top_level_import_kmeans_dba():
    from polars_ts import kmeans_dba as fn

    assert callable(fn)


def test_top_level_import_timeseries_kmeans():
    from polars_ts import TimeSeriesKMeans as cls

    assert cls is not None


def test_clustering_module_import():
    from polars_ts.clustering import TimeSeriesKMeans as cls
    from polars_ts.clustering import kmeans_dba as fn

    assert callable(fn)
    assert cls is not None
