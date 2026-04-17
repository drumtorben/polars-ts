import polars as pl
import pytest

from polars_ts.clustering import TimeSeriesKMedoids


@pytest.fixture
def cluster_data():
    """Create data with two clear clusters: ascending and descending."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
            "y": (
                [1.0, 2.0, 3.0, 4.0]  # ascending A
                + [1.0, 2.0, 3.0, 4.5]  # ascending B (similar to A)
                + [4.0, 3.0, 2.0, 1.0]  # descending C
                + [4.5, 3.0, 2.0, 1.0]  # descending D (similar to C)
            ),
        }
    )


class TestTimeSeriesKMedoids:
    def test_fit_returns_self(self, cluster_data):
        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        result = km.fit(cluster_data)
        assert result is km

    def test_labels_shape(self, cluster_data):
        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        km.fit(cluster_data)
        assert km.labels_ is not None
        assert km.labels_.shape[0] == 4
        assert "unique_id" in km.labels_.columns
        assert "cluster" in km.labels_.columns

    def test_two_clusters(self, cluster_data):
        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        km.fit(cluster_data)
        labels = dict(zip(km.labels_["unique_id"].to_list(), km.labels_["cluster"].to_list(), strict=False))
        # A and B should be in the same cluster, C and D in another
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_medoids_set(self, cluster_data):
        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        km.fit(cluster_data)
        assert len(km.medoids_) == 2

    def test_single_cluster(self, cluster_data):
        km = TimeSeriesKMedoids(n_clusters=1, metric="dtw")
        km.fit(cluster_data)
        labels = km.labels_["cluster"].to_list()
        assert all(label == 0 for label in labels)

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        km = TimeSeriesKMedoids(n_clusters=5)
        with pytest.raises(ValueError, match="Cannot create"):
            km.fit(df)

    def test_different_metrics(self, cluster_data):
        for metric in ["dtw", "erp", "lcss"]:
            km = TimeSeriesKMedoids(n_clusters=2, metric=metric)
            km.fit(cluster_data)
            assert km.labels_ is not None

    def test_identical_series_same_cluster(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0],
            }
        )
        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        km.fit(df)
        labels = dict(zip(km.labels_["unique_id"].to_list(), km.labels_["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
