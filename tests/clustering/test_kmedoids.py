import polars as pl
import pytest

from polars_ts.clustering.kmedoids import kmedoids


@pytest.fixture
def clusterable_series():
    """Three pairs of similar series: (A,B) close, (C,D) close, (E,F) close."""
    return pl.DataFrame(
        {
            "unique_id": (["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5),
            "y": (
                [1.0, 1.0, 1.0, 1.0, 1.0]  # A - flat low
                + [1.0, 1.0, 1.0, 1.0, 1.1]  # B - flat low (close to A)
                + [5.0, 5.0, 5.0, 5.0, 5.0]  # C - flat high
                + [5.0, 5.0, 5.0, 5.0, 5.1]  # D - flat high (close to C)
                + [1.0, 5.0, 1.0, 5.0, 1.0]  # E - oscillating
                + [1.0, 5.0, 1.0, 5.0, 1.1]  # F - oscillating (close to E)
            ),
        }
    )


class TestKmedoidsOutput:
    def test_returns_correct_columns(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        assert result.columns == ["unique_id", "cluster"]

    def test_k_clusters_assigned(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        n_clusters = result["cluster"].n_unique()
        assert n_clusters == 3

    def test_every_id_assigned(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        input_ids = set(clusterable_series["unique_id"].unique().to_list())
        output_ids = set(result["unique_id"].to_list())
        assert input_ids == output_ids

    def test_cluster_labels_are_zero_indexed(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        labels = sorted(result["cluster"].unique().to_list())
        assert labels == [0, 1, 2]


class TestKmedoidsCorrectness:
    def test_similar_series_same_cluster(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        cluster_map = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        # A and B should be in the same cluster
        assert cluster_map["A"] == cluster_map["B"]
        # C and D should be in the same cluster
        assert cluster_map["C"] == cluster_map["D"]
        # E and F should be in the same cluster
        assert cluster_map["E"] == cluster_map["F"]

    def test_dissimilar_series_different_clusters(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3)
        cluster_map = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        # The three groups should be in different clusters
        assert len({cluster_map["A"], cluster_map["C"], cluster_map["E"]}) == 3

    def test_deterministic_with_seed(self, clusterable_series):
        r1 = kmedoids(clusterable_series, k=3, seed=42)
        r2 = kmedoids(clusterable_series, k=3, seed=42)
        assert r1["cluster"].to_list() == r2["cluster"].to_list()

    def test_different_seed_may_differ(self, clusterable_series):
        # Just test it runs; different seeds might or might not give different results
        r1 = kmedoids(clusterable_series, k=3, seed=1)
        r2 = kmedoids(clusterable_series, k=3, seed=999)
        assert r1.shape == r2.shape


class TestKmedoidsEdgeCases:
    def test_k_equals_1(self, clusterable_series):
        result = kmedoids(clusterable_series, k=1)
        assert result["cluster"].n_unique() == 1

    def test_k_equals_n(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        result = kmedoids(df, k=3)
        assert result["cluster"].n_unique() == 3

    def test_k_too_large_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 3 + ["B"] * 3, "y": [1.0, 2.0, 3.0] * 2})
        with pytest.raises(ValueError, match="k .* must be <= number of series"):
            kmedoids(df, k=5)

    def test_k_zero_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 3, "y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="k must be >= 1"):
            kmedoids(df, k=0)

    def test_int_ids(self):
        df = pl.DataFrame(
            {
                "unique_id": [1] * 4 + [2] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.1],
            }
        )
        result = kmedoids(df, k=1)
        assert result["unique_id"].dtype == pl.Int64

    def test_invalid_method_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 3 + ["B"] * 3, "y": [1.0, 2.0, 3.0] * 2})
        with pytest.raises(ValueError, match="Unknown distance method"):
            kmedoids(df, k=1, method="nonexistent")


class TestKmedoidsMetrics:
    def test_with_erp(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3, method="erp")
        assert result.shape[0] == 6

    def test_with_lcss(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3, method="lcss", epsilon=1.0)
        assert result.shape[0] == 6

    def test_with_msm(self, clusterable_series):
        result = kmedoids(clusterable_series, k=3, method="msm")
        assert result.shape[0] == 6
