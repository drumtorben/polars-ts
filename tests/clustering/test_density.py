import polars as pl
import pytest

from polars_ts.clustering.density import dbscan_cluster, hdbscan_cluster


@pytest.fixture
def well_separated_data():
    """Six series in two well-separated groups (ascending vs descending)."""
    ascending = [1.0, 2.0, 3.0, 4.0]
    descending = [4.0, 3.0, 2.0, 1.0]
    return pl.DataFrame(
        {
            "unique_id": (["A1"] * 4 + ["A2"] * 4 + ["A3"] * 4 + ["B1"] * 4 + ["B2"] * 4 + ["B3"] * 4),
            "y": (
                ascending
                + [1.0, 2.1, 3.0, 4.1]
                + [1.0, 1.9, 3.1, 4.0]
                + descending
                + [4.1, 3.0, 2.0, 0.9]
                + [3.9, 3.1, 1.9, 1.0]
            ),
        }
    )


class TestHDBSCAN:
    def test_schema(self, well_separated_data):
        result = hdbscan_cluster(well_separated_data, method="dtw", min_cluster_size=2)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 6

    def test_cluster_dtype(self, well_separated_data):
        result = hdbscan_cluster(well_separated_data, method="dtw", min_cluster_size=2)
        assert result["cluster"].dtype == pl.Int64

    def test_finds_clusters(self, well_separated_data):
        result = hdbscan_cluster(well_separated_data, method="dtw", min_cluster_size=2, min_samples=1)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        # At least some points should be clustered (not all noise)
        non_noise = {k: v for k, v in labels.items() if v != -1}
        assert len(non_noise) >= 4, f"Expected clustered points, got {labels}"
        # A-series should share a cluster, B-series should share a cluster
        a_labels = {labels[k] for k in ["A1", "A2", "A3"] if labels[k] != -1}
        b_labels = {labels[k] for k in ["B1", "B2", "B3"] if labels[k] != -1}
        assert a_labels and b_labels, f"Both groups should have clustered points: {labels}"
        assert a_labels.isdisjoint(b_labels)

    def test_noise_label(self):
        """With dissimilar series and high min_cluster_size, everything is noise."""
        df = pl.DataFrame(
            {
                "unique_id": (["X"] * 4 + ["Y"] * 4 + ["Z"] * 4 + ["W"] * 4 + ["V"] * 4 + ["U"] * 4),
                "y": (
                    [1.0, 2.0, 3.0, 4.0]
                    + [10.0, 20.0, 30.0, 40.0]
                    + [100.0, 200.0, 300.0, 400.0]
                    + [5.0, 15.0, 25.0, 35.0]
                    + [50.0, 150.0, 250.0, 350.0]
                    + [500.0, 1500.0, 2500.0, 3500.0]
                ),
            }
        )
        result = hdbscan_cluster(df, method="dtw", min_cluster_size=5, min_samples=3)
        assert all(c == -1 for c in result["cluster"].to_list())

    def test_different_metrics(self, well_separated_data):
        for metric in ["dtw", "erp", "lcss"]:
            result = hdbscan_cluster(well_separated_data, method=metric, min_cluster_size=2)
            assert result.shape[0] == 6

    def test_min_samples_parameter(self, well_separated_data):
        result = hdbscan_cluster(well_separated_data, method="dtw", min_cluster_size=2, min_samples=1)
        assert result.shape[0] == 6

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.1, 3.0, 4.1, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = hdbscan_cluster(df, method="dtw", min_cluster_size=2, id_col="ts_id", target_col="value")
        assert "ts_id" in result.columns
        assert "cluster" in result.columns

    def test_top_level_import(self):
        from polars_ts import hdbscan_cluster as hc

        assert callable(hc)


class TestDBSCAN:
    def test_schema(self, well_separated_data):
        result = dbscan_cluster(well_separated_data, method="dtw", eps=2.0, min_samples=2)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 6

    def test_cluster_dtype(self, well_separated_data):
        result = dbscan_cluster(well_separated_data, method="dtw", eps=2.0, min_samples=2)
        assert result["cluster"].dtype == pl.Int64

    def test_finds_clusters(self, well_separated_data):
        result = dbscan_cluster(well_separated_data, method="dtw", eps=2.0, min_samples=2)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        non_noise = {k: v for k, v in labels.items() if v != -1}
        assert len(non_noise) >= 4, f"Expected clustered points, got {labels}"
        a_labels = {labels[k] for k in ["A1", "A2", "A3"] if labels[k] != -1}
        b_labels = {labels[k] for k in ["B1", "B2", "B3"] if labels[k] != -1}
        assert a_labels and b_labels, f"Both groups should have clustered points: {labels}"
        assert a_labels.isdisjoint(b_labels)

    def test_all_noise(self):
        """With tiny eps, all points become noise."""
        df = pl.DataFrame(
            {
                "unique_id": ["X"] * 4 + ["Y"] * 4 + ["Z"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
            }
        )
        result = dbscan_cluster(df, method="dtw", eps=0.001, min_samples=2)
        assert all(c == -1 for c in result["cluster"].to_list())

    def test_eps_controls_granularity(self, well_separated_data):
        """Larger eps should produce fewer or equal clusters (more merging)."""
        result_small = dbscan_cluster(well_separated_data, method="dtw", eps=0.5, min_samples=2)
        result_large = dbscan_cluster(well_separated_data, method="dtw", eps=100.0, min_samples=2)
        n_small = result_small["cluster"].filter(result_small["cluster"] >= 0).n_unique()
        n_large = result_large["cluster"].filter(result_large["cluster"] >= 0).n_unique()
        assert n_large <= n_small or n_large <= 1

    def test_different_metrics(self, well_separated_data):
        for metric in ["dtw", "erp", "lcss"]:
            result = dbscan_cluster(well_separated_data, method=metric, eps=2.0, min_samples=2)
            assert result.shape[0] == 6

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.1, 3.0, 4.1, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = dbscan_cluster(df, method="dtw", eps=2.0, min_samples=2, id_col="ts_id", target_col="value")
        assert "ts_id" in result.columns
        assert "cluster" in result.columns

    def test_top_level_import(self):
        from polars_ts import dbscan_cluster as dc

        assert callable(dc)
