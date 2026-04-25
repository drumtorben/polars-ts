import importlib

import polars as pl
import pytest

from polars_ts.clustering.auto import AutoClusterResult, auto_cluster

_has_scipy = importlib.util.find_spec("scipy") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None


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


class TestAutoClusterResult:
    def test_returns_auto_cluster_result(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kmedoids"], distances=["sbd"], k_range=range(2, 4))
        assert isinstance(result, AutoClusterResult)

    def test_result_has_best_labels(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kmedoids"], distances=["sbd"], k_range=range(2, 4))
        assert isinstance(result.best_labels, pl.DataFrame)
        assert "unique_id" in result.best_labels.columns
        assert "cluster" in result.best_labels.columns
        assert result.best_labels.shape[0] == 6

    def test_result_has_metadata(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kmedoids"], distances=["sbd"], k_range=range(2, 4))
        assert isinstance(result.best_method, str)
        assert isinstance(result.best_distance, str)
        assert isinstance(result.best_k, int)
        assert isinstance(result.best_score, float)

    def test_result_has_results_table(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kmedoids"], distances=["sbd"], k_range=range(2, 4))
        assert isinstance(result.results_table, pl.DataFrame)
        assert "method" in result.results_table.columns
        assert "distance" in result.results_table.columns
        assert "k" in result.results_table.columns
        assert "score" in result.results_table.columns
        # kmedoids x sbd x k=[2,3] => 2 rows
        assert result.results_table.shape[0] == 2


class TestAutoClusterMethods:
    def test_kmedoids_method(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kmedoids"], distances=["sbd"], k_range=range(2, 3))
        assert result.best_method == "kmedoids"
        assert result.best_k == 2

    @pytest.mark.skipif(not _has_scipy or not _has_sklearn, reason="scipy/sklearn required")
    def test_spectral_method(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["spectral"], distances=["sbd"], k_range=range(2, 3))
        assert result.best_method == "spectral"

    def test_kshape_method(self, well_separated_data):
        result = auto_cluster(well_separated_data, methods=["kshape"], distances=["sbd"], k_range=range(2, 3))
        assert result.best_method == "kshape"

    def test_kshape_skips_non_sbd(self, well_separated_data):
        """Kshape only works with SBD; non-SBD combos should be skipped."""
        result = auto_cluster(
            well_separated_data,
            methods=["kshape"],
            distances=["dtw", "sbd"],
            k_range=range(2, 3),
        )
        # Only SBD rows should appear in results
        assert all(row["distance"] == "sbd" for row in result.results_table.to_dicts())

    @pytest.mark.skipif(not _has_sklearn, reason="sklearn required")
    def test_hdbscan_no_k(self, well_separated_data):
        """HDBSCAN doesn't use k; should produce one row per distance."""
        result = auto_cluster(
            well_separated_data,
            methods=["hdbscan"],
            distances=["sbd"],
            k_range=range(2, 4),
            hdbscan_kwargs={"min_cluster_size": 2},
        )
        # Should have exactly 1 row (no k iteration)
        assert result.results_table.shape[0] == 1
        assert result.results_table["k"][0] is None

    @pytest.mark.skipif(not _has_sklearn, reason="sklearn required")
    def test_dbscan_no_k(self, well_separated_data):
        """DBSCAN doesn't use k; should produce one row per distance."""
        result = auto_cluster(
            well_separated_data,
            methods=["dbscan"],
            distances=["sbd"],
            k_range=range(2, 4),
            dbscan_kwargs={"eps": 0.1, "min_samples": 2},
        )
        assert result.results_table.shape[0] == 1
        assert result.results_table["k"][0] is None


class TestAutoClusterMultipleCombinations:
    @pytest.mark.skipif(not _has_scipy or not _has_sklearn, reason="scipy/sklearn required")
    def test_multiple_methods_and_distances(self, well_separated_data):
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids", "spectral"],
            distances=["sbd", "dtw"],
            k_range=range(2, 4),
        )
        # 2 methods x 2 distances x 2 k values = 8 rows
        assert result.results_table.shape[0] == 8

    @pytest.mark.skipif(not _has_scipy or not _has_sklearn, reason="scipy/sklearn required")
    def test_best_is_optimal(self, well_separated_data):
        """Best score should be the max silhouette in the results table."""
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids", "spectral"],
            distances=["sbd"],
            k_range=range(2, 4),
            metric="silhouette",
        )
        max_score = result.results_table["score"].max()
        assert result.best_score == pytest.approx(max_score)


class TestAutoClusterMetrics:
    def test_silhouette_metric(self, well_separated_data):
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 3),
            metric="silhouette",
        )
        assert result.best_score > 0  # well-separated data should have positive silhouette

    def test_davies_bouldin_metric(self, well_separated_data):
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 3),
            metric="davies_bouldin",
        )
        assert result.best_score >= 0

    def test_calinski_harabasz_metric(self, well_separated_data):
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 3),
            metric="calinski_harabasz",
        )
        assert result.best_score > 0

    def test_invalid_metric_raises(self, well_separated_data):
        with pytest.raises(ValueError, match="Unknown metric"):
            auto_cluster(
                well_separated_data,
                methods=["kmedoids"],
                distances=["sbd"],
                k_range=range(2, 3),
                metric="unknown",
            )


class TestAutoClusterEdgeCases:
    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "value": [1.0, 2.0, 3.0, 4.0, 1.0, 2.1, 3.0, 4.1, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = auto_cluster(
            df,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 3),
            id_col="ts_id",
            target_col="value",
        )
        assert "ts_id" in result.best_labels.columns

    def test_single_method_single_k(self, well_separated_data):
        result = auto_cluster(
            well_separated_data,
            methods=["kmedoids"],
            distances=["sbd"],
            k_range=range(2, 3),
        )
        assert result.results_table.shape[0] == 1

    def test_all_invalid_combos_raises(self):
        """If all combinations are invalid (e.g. kshape + dtw only), raise."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.1, 3.0, 4.1, 4.0, 3.0, 2.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="No valid"):
            auto_cluster(df, methods=["kshape"], distances=["dtw"], k_range=range(2, 3))


class TestAutoClusterImports:
    def test_top_level_import(self):
        from polars_ts import auto_cluster as ac

        assert callable(ac)

    def test_clustering_import(self):
        from polars_ts.clustering import auto_cluster as ac

        assert callable(ac)
