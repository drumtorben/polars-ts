import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.hierarchical import agglomerative_cluster


@pytest.fixture
def cluster_data():
    """Four series in two well-separated groups (ascending vs descending)."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
            "y": ([1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 3.0, 4.5] + [4.0, 3.0, 2.0, 1.0] + [4.5, 3.0, 2.0, 1.0]),
        }
    )


class TestAgglomerativeCluster:
    def test_schema(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 4

    def test_cluster_dtype(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2)
        assert result["cluster"].dtype == pl.Int64

    def test_two_clusters(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_single_cluster(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=1)
        assert all(c == 0 for c in result["cluster"].to_list())

    def test_n_equals_series(self, cluster_data):
        """Each series in its own cluster."""
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=4)
        labels = result["cluster"].to_list()
        assert len(set(labels)) == 4

    def test_zero_based_labels(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=3)
        labels = result["cluster"].to_list()
        assert min(labels) == 0

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="n_clusters"):
            agglomerative_cluster(df, n_clusters=5)

    def test_zero_clusters_raises(self, cluster_data):
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            agglomerative_cluster(cluster_data, n_clusters=0)

    def test_invalid_linkage_raises(self, cluster_data):
        with pytest.raises(ValueError, match="Unknown linkage"):
            agglomerative_cluster(cluster_data, n_clusters=2, linkage_method="invalid")


class TestLinkageMethods:
    def test_average(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, linkage_method="average")
        assert result.shape[0] == 4

    def test_complete(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, linkage_method="complete")
        assert result.shape[0] == 4

    def test_single(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, linkage_method="single")
        assert result.shape[0] == 4

    def test_weighted(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, linkage_method="weighted")
        assert result.shape[0] == 4


class TestReturnLinkage:
    def test_returns_tuple(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, return_linkage=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_linkage_matrix_shape(self, cluster_data):
        _, Z = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, return_linkage=True)
        # Linkage matrix has shape (n-1, 4)
        assert isinstance(Z, np.ndarray)
        assert Z.shape == (3, 4)

    def test_linkage_matrix_six_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4 + ["E"] * 4 + ["F"] * 4,
                "y": (
                    [1.0, 2.0, 3.0, 4.0]
                    + [1.0, 2.1, 3.0, 4.1]
                    + [1.0, 1.9, 3.1, 4.0]
                    + [4.0, 3.0, 2.0, 1.0]
                    + [4.1, 3.0, 2.0, 0.9]
                    + [3.9, 3.1, 1.9, 1.0]
                ),
            }
        )
        _, Z = agglomerative_cluster(df, method="dtw", n_clusters=2, return_linkage=True)
        assert Z.shape == (5, 4)

    def test_labels_same_with_and_without_linkage(self, cluster_data):
        labels_only = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2)
        labels_with, _ = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, return_linkage=True)
        assert labels_only["cluster"].to_list() == labels_with["cluster"].to_list()

    def test_without_return_linkage_returns_dataframe(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="dtw", n_clusters=2, return_linkage=False)
        assert isinstance(result, pl.DataFrame)


class TestDistanceMetrics:
    def test_erp(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="erp", n_clusters=2)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]

    def test_lcss(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="lcss", n_clusters=2)
        assert result.shape[0] == 4

    def test_sbd(self, cluster_data):
        result = agglomerative_cluster(cluster_data, method="sbd", n_clusters=2)
        assert result.shape[0] == 4


def test_custom_columns():
    df = pl.DataFrame(
        {
            "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.5, 4.0, 3.0, 2.0, 1.0],
        }
    )
    result = agglomerative_cluster(df, method="dtw", n_clusters=2, id_col="ts_id", target_col="value")
    assert "ts_id" in result.columns
    assert "cluster" in result.columns
    assert len(result) == 3


def test_top_level_import():
    from polars_ts import agglomerative_cluster as ac

    assert callable(ac)
