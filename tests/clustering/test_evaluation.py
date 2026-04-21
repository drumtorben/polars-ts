import polars as pl
import pytest

from polars_ts.clustering.evaluation import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


@pytest.fixture
def two_cluster_data():
    """Two clear clusters: ascending vs descending."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
            "y": ([1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 3.0, 4.5] + [4.0, 3.0, 2.0, 1.0] + [4.5, 3.0, 2.0, 1.0]),
        }
    )


@pytest.fixture
def two_cluster_labels():
    return pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 0, 1, 1]})


@pytest.fixture
def bad_labels():
    """Wrong clustering: mix ascending and descending in same cluster."""
    return pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 1, 0, 1]})


@pytest.fixture
def three_series_data():
    return pl.DataFrame(
        {
            "unique_id": ["X"] * 3 + ["Y"] * 3 + ["Z"] * 3,
            "y": [1.0, 2.0, 3.0] + [1.0, 2.0, 3.5] + [3.0, 2.0, 1.0],
        }
    )


# --- Silhouette Score ---


class TestSilhouetteScore:
    def test_good_clustering_positive(self, two_cluster_data, two_cluster_labels):
        score = silhouette_score(two_cluster_data, two_cluster_labels, method="dtw")
        assert score > 0, f"Expected positive silhouette for good clustering, got {score}"

    def test_bad_clustering_lower(self, two_cluster_data, two_cluster_labels, bad_labels):
        good = silhouette_score(two_cluster_data, two_cluster_labels, method="dtw")
        bad = silhouette_score(two_cluster_data, bad_labels, method="dtw")
        assert good > bad, f"Good clustering ({good}) should score higher than bad ({bad})"

    def test_range(self, two_cluster_data, two_cluster_labels):
        score = silhouette_score(two_cluster_data, two_cluster_labels, method="dtw")
        assert -1.0 <= score <= 1.0

    def test_single_cluster_returns_zero(self, two_cluster_data):
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 0, 0, 0]})
        assert silhouette_score(two_cluster_data, labels, method="dtw") == 0.0

    def test_single_sample(self):
        df = pl.DataFrame({"unique_id": ["A"] * 3, "y": [1.0, 2.0, 3.0]})
        labels = pl.DataFrame({"unique_id": ["A"], "cluster": [0]})
        assert silhouette_score(df, labels, method="dtw") == 0.0

    def test_different_metrics(self, two_cluster_data, two_cluster_labels):
        for metric in ["dtw", "msm", "erp"]:
            score = silhouette_score(two_cluster_data, two_cluster_labels, method=metric)
            assert -1.0 <= score <= 1.0

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.5, 3.0, 2.0, 1.0],
            }
        )
        labels = pl.DataFrame({"ts_id": ["A", "B", "C"], "cluster": [0, 0, 1]})
        score = silhouette_score(df, labels, method="dtw", id_col="ts_id", target_col="value")
        assert -1.0 <= score <= 1.0


# --- Silhouette Samples ---


class TestSilhouetteSamples:
    def test_output_shape(self, two_cluster_data, two_cluster_labels):
        result = silhouette_samples(two_cluster_data, two_cluster_labels, method="dtw")
        assert result.shape[0] == 4
        assert set(result.columns) == {"unique_id", "cluster", "silhouette"}

    def test_values_in_range(self, two_cluster_data, two_cluster_labels):
        result = silhouette_samples(two_cluster_data, two_cluster_labels, method="dtw")
        for val in result["silhouette"].to_list():
            assert -1.0 <= val <= 1.0

    def test_mean_matches_score(self, two_cluster_data, two_cluster_labels):
        samples = silhouette_samples(two_cluster_data, two_cluster_labels, method="dtw")
        mean_from_samples = samples["silhouette"].mean()
        score = silhouette_score(two_cluster_data, two_cluster_labels, method="dtw")
        assert abs(mean_from_samples - score) < 1e-10

    def test_single_cluster(self, two_cluster_data):
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 0, 0, 0]})
        result = silhouette_samples(two_cluster_data, labels, method="dtw")
        assert all(v == 0.0 for v in result["silhouette"].to_list())

    def test_cluster_column_preserved(self, two_cluster_data, two_cluster_labels):
        result = silhouette_samples(two_cluster_data, two_cluster_labels, method="dtw")
        cluster_map = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        label_map = dict(
            zip(
                two_cluster_labels["unique_id"].to_list(),
                two_cluster_labels["cluster"].to_list(),
                strict=False,
            )
        )
        for uid in cluster_map:
            assert cluster_map[uid] == label_map[uid]


# --- Davies-Bouldin ---


class TestDaviesBouldinScore:
    def test_good_clustering_lower(self, two_cluster_data, two_cluster_labels, bad_labels):
        good = davies_bouldin_score(two_cluster_data, two_cluster_labels, method="dtw")
        bad = davies_bouldin_score(two_cluster_data, bad_labels, method="dtw")
        assert good < bad, f"Good clustering ({good}) should have lower DB than bad ({bad})"

    def test_non_negative(self, two_cluster_data, two_cluster_labels):
        score = davies_bouldin_score(two_cluster_data, two_cluster_labels, method="dtw")
        assert score >= 0.0

    def test_single_cluster_returns_zero(self, two_cluster_data):
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 0, 0, 0]})
        assert davies_bouldin_score(two_cluster_data, labels, method="dtw") == 0.0

    def test_different_metrics(self, two_cluster_data, two_cluster_labels):
        for metric in ["dtw", "msm", "erp"]:
            score = davies_bouldin_score(two_cluster_data, two_cluster_labels, method=metric)
            assert score >= 0.0

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.5, 3.0, 2.0, 1.0],
            }
        )
        labels = pl.DataFrame({"ts_id": ["A", "B", "C"], "cluster": [0, 0, 1]})
        score = davies_bouldin_score(df, labels, method="dtw", id_col="ts_id", target_col="value")
        assert score >= 0.0


# --- Calinski-Harabasz ---


class TestCalinskiHarabaszScore:
    def test_good_clustering_higher(self, two_cluster_data, two_cluster_labels, bad_labels):
        good = calinski_harabasz_score(two_cluster_data, two_cluster_labels, method="dtw")
        bad = calinski_harabasz_score(two_cluster_data, bad_labels, method="dtw")
        assert good > bad, f"Good clustering ({good}) should have higher CH than bad ({bad})"

    def test_non_negative(self, two_cluster_data, two_cluster_labels):
        score = calinski_harabasz_score(two_cluster_data, two_cluster_labels, method="dtw")
        assert score >= 0.0

    def test_single_cluster_returns_zero(self, two_cluster_data):
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 0, 0, 0]})
        assert calinski_harabasz_score(two_cluster_data, labels, method="dtw") == 0.0

    def test_n_equals_k_returns_zero(self):
        """When every series is its own cluster, n == k, denominator is zero."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "y": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )
        labels = pl.DataFrame({"unique_id": ["A", "B"], "cluster": [0, 1]})
        assert calinski_harabasz_score(df, labels, method="dtw") == 0.0

    def test_different_metrics(self, two_cluster_data, two_cluster_labels):
        for metric in ["dtw", "msm", "erp"]:
            score = calinski_harabasz_score(two_cluster_data, two_cluster_labels, method=metric)
            assert score >= 0.0

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
                "value": ([1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 3.0, 4.5] + [4.0, 3.0, 2.0, 1.0] + [4.5, 3.0, 2.0, 1.0]),
            }
        )
        labels = pl.DataFrame({"ts_id": ["A", "B", "C", "D"], "cluster": [0, 0, 1, 1]})
        score = calinski_harabasz_score(df, labels, method="dtw", id_col="ts_id", target_col="value")
        assert score > 0.0


# --- Integration with clustering ---


class TestIntegrationWithKMedoids:
    def test_evaluate_kmedoids_result(self, two_cluster_data):
        from polars_ts.clustering.kmedoids import kmedoids

        labels = kmedoids(two_cluster_data, k=2, method="dtw")
        sil = silhouette_score(two_cluster_data, labels, method="dtw")
        db = davies_bouldin_score(two_cluster_data, labels, method="dtw")
        ch = calinski_harabasz_score(two_cluster_data, labels, method="dtw")

        assert -1.0 <= sil <= 1.0
        assert db >= 0.0
        assert ch >= 0.0

    def test_evaluate_kmedoids_class(self, two_cluster_data):
        from polars_ts.clustering.kmedoids import TimeSeriesKMedoids

        km = TimeSeriesKMedoids(n_clusters=2, metric="dtw")
        km.fit(two_cluster_data)
        assert km.labels_ is not None

        sil = silhouette_score(two_cluster_data, km.labels_, method="dtw")
        assert sil > 0


# --- Edge cases ---


class TestEdgeCases:
    def test_all_singleton_clusters_silhouette(self, two_cluster_data):
        """Each series in its own cluster — a_i=0, b_i>0, so silhouette=1.0."""
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 1, 2, 3]})
        score = silhouette_score(two_cluster_data, labels, method="dtw")
        assert score == 1.0

    def test_all_singleton_clusters_davies_bouldin(self, two_cluster_data):
        labels = pl.DataFrame({"unique_id": ["A", "B", "C", "D"], "cluster": [0, 1, 2, 3]})
        score = davies_bouldin_score(two_cluster_data, labels, method="dtw")
        assert score >= 0.0

    def test_constant_series_mase(self):
        """Constant series: naive MAE = 0, forecast MAE > 0 → MASE = inf."""
        from polars_ts.metrics.forecast import mase

        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5,
                "ds": list(range(5)),
                "y": [5.0, 5.0, 5.0, 5.0, 5.0],
                "y_hat": [5.0, 5.0, 5.0, 5.0, 6.0],
            }
        )
        result = mase(df, season_length=1)
        assert result == float("inf")

    def test_all_zero_actuals_mape(self):
        """All-zero actuals: MAPE excludes all rows, returns None."""
        from polars_ts.metrics.forecast import mape

        df = pl.DataFrame({"y": [0.0, 0.0], "y_hat": [1.0, 2.0]})
        result = mape(df)
        assert result is None

    def test_crps_unparseable_columns(self):
        """Quantile column names that can't be parsed should raise ValueError."""
        from polars_ts.metrics.forecast import crps

        df = pl.DataFrame({"y": [10.0], "q_low": [8.0]})
        with pytest.raises(ValueError, match="Cannot parse quantile levels"):
            crps(df)


def test_cross_metric_consistency(two_cluster_data, two_cluster_labels, bad_labels):
    """Good clustering should be better by all metrics simultaneously."""
    good_sil = silhouette_score(two_cluster_data, two_cluster_labels, method="dtw")
    bad_sil = silhouette_score(two_cluster_data, bad_labels, method="dtw")
    good_db = davies_bouldin_score(two_cluster_data, two_cluster_labels, method="dtw")
    bad_db = davies_bouldin_score(two_cluster_data, bad_labels, method="dtw")
    good_ch = calinski_harabasz_score(two_cluster_data, two_cluster_labels, method="dtw")
    bad_ch = calinski_harabasz_score(two_cluster_data, bad_labels, method="dtw")
    # Good clustering: higher silhouette, lower DB, higher CH
    assert good_sil > bad_sil
    assert good_db < bad_db
    assert good_ch > bad_ch


class TestThreeClusterMetrics:
    """Metrics with 3+ clusters."""

    @pytest.fixture
    def three_cluster_data(self):
        return pl.DataFrame(
            {
                "unique_id": (["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4 + ["E"] * 4 + ["F"] * 4),
                "y": (
                    [1.0, 2.0, 3.0, 4.0]  # ascending A
                    + [1.0, 2.0, 3.0, 4.5]  # ascending B
                    + [4.0, 3.0, 2.0, 1.0]  # descending C
                    + [4.5, 3.0, 2.0, 1.0]  # descending D
                    + [2.0, 2.0, 2.0, 2.0]  # flat E
                    + [2.1, 2.0, 2.1, 2.0]  # flat F
                ),
            }
        )

    @pytest.fixture
    def three_cluster_labels(self):
        return pl.DataFrame(
            {
                "unique_id": ["A", "B", "C", "D", "E", "F"],
                "cluster": [0, 0, 1, 1, 2, 2],
            }
        )

    def test_silhouette_three_clusters(self, three_cluster_data, three_cluster_labels):
        score = silhouette_score(three_cluster_data, three_cluster_labels, method="dtw")
        assert -1.0 <= score <= 1.0

    def test_davies_bouldin_three_clusters(self, three_cluster_data, three_cluster_labels):
        score = davies_bouldin_score(three_cluster_data, three_cluster_labels, method="dtw")
        assert score >= 0.0

    def test_calinski_harabasz_three_clusters(self, three_cluster_data, three_cluster_labels):
        score = calinski_harabasz_score(three_cluster_data, three_cluster_labels, method="dtw")
        assert score > 0.0

    def test_silhouette_samples_three_clusters(self, three_cluster_data, three_cluster_labels):
        result = silhouette_samples(three_cluster_data, three_cluster_labels, method="dtw")
        assert result.shape[0] == 6
        assert result["cluster"].n_unique() == 3
