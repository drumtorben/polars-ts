import polars as pl
import pytest

from polars_ts.clustering.scalable import clara, clarans

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
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                + [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
                + [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                + [6.1, 5.1, 4.1, 3.1, 2.1, 1.1]
            ),
        }
    )


@pytest.fixture
def six_series_data():
    """Six series in three groups for k=3 tests."""
    return pl.DataFrame(
        {
            "unique_id": (["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8 + ["E"] * 8 + ["F"] * 8),
            "y": (
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                + [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]
                + [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                + [8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1]
                + [1.0, 8.0, 1.0, 8.0, 1.0, 8.0, 1.0, 8.0]
                + [1.1, 7.9, 1.1, 7.9, 1.1, 7.9, 1.1, 7.9]
            ),
        }
    )


# ---------------------------------------------------------------------------
# CLARA tests
# ---------------------------------------------------------------------------


class TestClara:
    def test_returns_dataframe(self, cluster_data):
        result = clara(cluster_data, k=2, n_samples=2, sample_size=4)
        assert isinstance(result, pl.DataFrame)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 4

    def test_correct_clustering(self, cluster_data):
        result = clara(cluster_data, k=2, n_samples=3, sample_size=4, seed=42)
        labels = dict(
            zip(
                result["unique_id"].to_list(),
                result["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_three_clusters(self, six_series_data):
        result = clara(six_series_data, k=3, n_samples=3, sample_size=6, seed=42)
        assert result["cluster"].n_unique() == 3
        labels = dict(
            zip(
                result["unique_id"].to_list(),
                result["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["E"] == labels["F"]

    def test_sample_size_larger_than_n(self, cluster_data):
        """When sample_size >= n, should fall back to full PAM."""
        result = clara(cluster_data, k=2, n_samples=1, sample_size=100, seed=42)
        assert result.shape[0] == 4
        assert result["cluster"].n_unique() == 2

    def test_single_cluster(self, cluster_data):
        result = clara(cluster_data, k=1, n_samples=1, sample_size=4)
        labels = result["cluster"].to_list()
        assert all(label == 0 for label in labels)

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="must be <="):
            clara(df, k=5)

    def test_seed_reproducibility(self, cluster_data):
        r1 = clara(cluster_data, k=2, n_samples=3, sample_size=3, seed=123)
        r2 = clara(cluster_data, k=2, n_samples=3, sample_size=3, seed=123)
        assert r1["cluster"].to_list() == r2["cluster"].to_list()

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["X"] * 4 + ["Y"] * 4,
                "val": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = clara(df, k=2, n_samples=1, sample_size=2, id_col="ts_id", target_col="val")
        assert "ts_id" in result.columns

    def test_cluster_labels_contiguous(self, cluster_data):
        """Cluster labels should be 0-based contiguous integers."""
        result = clara(cluster_data, k=2, n_samples=2, sample_size=4, seed=42)
        labels = sorted(result["cluster"].unique().to_list())
        assert labels == [0, 1]


# ---------------------------------------------------------------------------
# CLARANS tests
# ---------------------------------------------------------------------------


class TestClarans:
    def test_returns_dataframe(self, cluster_data):
        result = clarans(cluster_data, k=2, num_local=2, max_neighbor=5)
        assert isinstance(result, pl.DataFrame)
        assert "unique_id" in result.columns
        assert "cluster" in result.columns
        assert result.shape[0] == 4

    def test_correct_clustering(self, cluster_data):
        result = clarans(cluster_data, k=2, num_local=3, max_neighbor=10, seed=42)
        labels = dict(
            zip(
                result["unique_id"].to_list(),
                result["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["A"] != labels["C"]

    def test_three_clusters(self, six_series_data):
        result = clarans(six_series_data, k=3, num_local=3, max_neighbor=20, seed=42)
        assert result["cluster"].n_unique() == 3
        labels = dict(
            zip(
                result["unique_id"].to_list(),
                result["cluster"].to_list(),
                strict=False,
            )
        )
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]
        assert labels["E"] == labels["F"]

    def test_single_cluster(self, cluster_data):
        result = clarans(cluster_data, k=1, num_local=1, max_neighbor=5)
        labels = result["cluster"].to_list()
        assert all(label == 0 for label in labels)

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="must be <="):
            clarans(df, k=5)

    def test_seed_reproducibility(self, cluster_data):
        r1 = clarans(cluster_data, k=2, num_local=2, max_neighbor=5, seed=99)
        r2 = clarans(cluster_data, k=2, num_local=2, max_neighbor=5, seed=99)
        assert r1["cluster"].to_list() == r2["cluster"].to_list()

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "ts_id": ["X"] * 4 + ["Y"] * 4,
                "val": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = clarans(df, k=2, num_local=1, max_neighbor=5, id_col="ts_id", target_col="val")
        assert "ts_id" in result.columns

    def test_cluster_labels_contiguous(self, cluster_data):
        result = clarans(cluster_data, k=2, num_local=2, max_neighbor=10, seed=42)
        labels = sorted(result["cluster"].unique().to_list())
        assert labels == [0, 1]


# ---------------------------------------------------------------------------
# Edge case and regression tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clara_k_equals_n(self):
        """When k == n, every series gets its own cluster."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0] + [5.0, 6.0, 7.0, 8.0] + [9.0, 10.0, 11.0, 12.0],
            }
        )
        result = clara(df, k=3, n_samples=1, sample_size=3)
        assert result["cluster"].n_unique() == 3

    def test_clarans_k_equals_n(self):
        """When k == n, every series is a medoid."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        result = clarans(df, k=2, num_local=1, max_neighbor=5)
        assert result["cluster"].n_unique() == 2

    def test_clara_sample_size_equals_k(self):
        """When sample_size == k, PAM gets exactly k series (minimal subsample)."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
                "y": ([1.0, 2.0, 3.0, 4.0] + [1.1, 2.1, 3.1, 4.1] + [4.0, 3.0, 2.0, 1.0] + [4.1, 3.1, 2.1, 1.1]),
            }
        )
        result = clara(df, k=2, n_samples=3, sample_size=2, seed=42)
        assert result.shape[0] == 4
        assert result["cluster"].n_unique() == 2
        labels = sorted(result["cluster"].unique().to_list())
        assert labels == [0, 1]

    def test_clarans_max_neighbor_zero(self):
        """With max_neighbor=0, no swaps happen — still returns valid result."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        result = clarans(df, k=2, num_local=1, max_neighbor=0)
        assert result.shape[0] == 2
        assert "cluster" in result.columns

    def test_clara_k_zero_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="must be >= 1"):
            clara(df, k=0)

    def test_clarans_k_zero_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 4, "y": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="must be >= 1"):
            clarans(df, k=0)

    def test_clara_two_series(self):
        """Minimal case: two series, k=2."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            }
        )
        result = clara(df, k=2, n_samples=1, sample_size=2)
        assert result["cluster"].n_unique() == 2
        labels = dict(zip(result["unique_id"].to_list(), result["cluster"].to_list(), strict=False))
        assert labels["A"] != labels["B"]

    def test_clarans_two_series(self):
        """Minimal case: two series, k=2."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 3,
                "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            }
        )
        result = clarans(df, k=2, num_local=1, max_neighbor=5)
        assert result["cluster"].n_unique() == 2

    def test_clara_output_preserves_id_dtype(self):
        """Output id column should have the same dtype as input."""
        df = pl.DataFrame(
            {
                "unique_id": pl.Series(["A"] * 4 + ["B"] * 4, dtype=pl.String),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        result = clara(df, k=2, n_samples=1, sample_size=2)
        assert result["unique_id"].dtype == pl.String


# ---------------------------------------------------------------------------
# Top-level import tests
# ---------------------------------------------------------------------------


def test_top_level_import_clara():
    from polars_ts import clara as fn

    assert callable(fn)


def test_top_level_import_clarans():
    from polars_ts import clarans as fn

    assert callable(fn)


def test_clustering_module_import():
    from polars_ts.clustering import clara as fn1
    from polars_ts.clustering import clarans as fn2

    assert callable(fn1)
    assert callable(fn2)
