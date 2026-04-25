import polars as pl
import pytest

scipy = pytest.importorskip("scipy")
sklearn = pytest.importorskip("sklearn")

from polars_ts.clustering.spectral import spectral_cluster  # noqa: E402


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


class TestSpectralCluster:
    def test_schema(self, well_separated_data):
        result = spectral_cluster(well_separated_data, k=2, method="sbd")
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 6

    def test_cluster_dtype(self, well_separated_data):
        result = spectral_cluster(well_separated_data, k=2, method="sbd")
        assert result["cluster"].dtype == pl.Int64

    def test_finds_clusters(self, well_separated_data):
        result = spectral_cluster(well_separated_data, k=2, method="sbd", sigma=1.0)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        a_labels = {labels[k] for k in ["A1", "A2", "A3"]}
        b_labels = {labels[k] for k in ["B1", "B2", "B3"]}
        # Each group should be in a single cluster and the two groups distinct
        assert len(a_labels) == 1, f"A-series should share a cluster: {labels}"
        assert len(b_labels) == 1, f"B-series should share a cluster: {labels}"
        assert a_labels.isdisjoint(b_labels), f"Groups should be in different clusters: {labels}"

    def test_dtw_method(self, well_separated_data):
        result = spectral_cluster(well_separated_data, k=2, method="dtw", sigma=1.0)
        assert result.shape[0] == 6
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        a_labels = {labels[k] for k in ["A1", "A2", "A3"]}
        b_labels = {labels[k] for k in ["B1", "B2", "B3"]}
        assert len(a_labels) == 1
        assert len(b_labels) == 1
        assert a_labels.isdisjoint(b_labels)

    def test_sigma_parameter(self, well_separated_data):
        """Different sigma values should still produce valid output."""
        for sigma in [0.1, 1.0, 10.0]:
            result = spectral_cluster(well_separated_data, k=2, method="sbd", sigma=sigma)
            assert result.shape[0] == 6
            assert result["cluster"].dtype == pl.Int64

    def test_single_cluster(self, well_separated_data):
        result = spectral_cluster(well_separated_data, k=1, method="sbd")
        assert result.shape[0] == 6
        assert result["cluster"].n_unique() == 1

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="Cannot create 5 clusters from 2"):
            spectral_cluster(df, k=5, method="sbd")

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.1, 3.0, 4.1, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = spectral_cluster(df, k=2, method="sbd", id_col="ts_id", target_col="value")
        assert "ts_id" in result.columns
        assert "cluster" in result.columns

    def test_seed_reproducibility(self, well_separated_data):
        r1 = spectral_cluster(well_separated_data, k=2, method="sbd", seed=123)
        r2 = spectral_cluster(well_separated_data, k=2, method="sbd", seed=123)
        assert r1["cluster"].to_list() == r2["cluster"].to_list()

    def test_three_clusters(self):
        """Three distinct patterns should be separated into three clusters."""
        flat = [2.0, 2.0, 2.0, 2.0]
        df = pl.DataFrame(
            {
                "unique_id": (
                    ["A1"] * 4
                    + ["A2"] * 4
                    + ["A3"] * 4
                    + ["B1"] * 4
                    + ["B2"] * 4
                    + ["B3"] * 4
                    + ["C1"] * 4
                    + ["C2"] * 4
                    + ["C3"] * 4
                ),
                "y": (
                    [1.0, 2.0, 3.0, 4.0]
                    + [1.0, 2.1, 3.0, 4.1]
                    + [1.0, 1.9, 3.1, 4.0]
                    + [4.0, 3.0, 2.0, 1.0]
                    + [4.1, 3.0, 2.0, 0.9]
                    + [3.9, 3.1, 1.9, 1.0]
                    + flat
                    + [2.0, 2.1, 2.0, 1.9]
                    + [2.1, 2.0, 1.9, 2.0]
                ),
            }
        )
        result = spectral_cluster(df, k=3, method="sbd", sigma=0.5)
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        a_labels = {labels[k] for k in ["A1", "A2", "A3"]}
        b_labels = {labels[k] for k in ["B1", "B2", "B3"]}
        c_labels = {labels[k] for k in ["C1", "C2", "C3"]}
        assert len(a_labels) == 1
        assert len(b_labels) == 1
        assert len(c_labels) == 1
        assert len(a_labels | b_labels | c_labels) == 3

    def test_top_level_import(self):
        from polars_ts import spectral_cluster as sc

        assert callable(sc)

    def test_clustering_import(self):
        from polars_ts.clustering import spectral_cluster as sc

        assert callable(sc)
