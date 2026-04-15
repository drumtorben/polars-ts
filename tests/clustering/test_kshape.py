import polars as pl
import pytest

from polars_ts.clustering import KShape


@pytest.fixture
def shape_data():
    """Create data with two distinct shape patterns."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8,
            "y": (
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]  # sine A
                + [0.0, 0.9, 0.0, -0.9, 0.0, 0.9, 0.0, -0.9]  # sine B
                + [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]  # cosine C
                + [0.9, 0.0, -0.9, 0.0, 0.9, 0.0, -0.9, 0.0]  # cosine D
            ),
        }
    )


class TestKShape:
    def test_fit_returns_self(self, shape_data):
        ks = KShape(n_clusters=2)
        result = ks.fit(shape_data)
        assert result is ks

    def test_labels_shape(self, shape_data):
        ks = KShape(n_clusters=2)
        ks.fit(shape_data)
        assert ks.labels_ is not None
        assert ks.labels_.shape[0] == 4
        assert "unique_id" in ks.labels_.columns
        assert "cluster" in ks.labels_.columns

    def test_centroids_count(self, shape_data):
        ks = KShape(n_clusters=2)
        ks.fit(shape_data)
        assert len(ks.centroids_) == 2

    def test_similar_shapes_grouped(self, shape_data):
        ks = KShape(n_clusters=2, max_iter=50)
        ks.fit(shape_data)
        labels = dict(zip(ks.labels_["unique_id"].to_list(), ks.labels_["cluster"].to_list(), strict=False))
        # A and B have same shape (sine), C and D have same shape (cosine)
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]

    def test_single_cluster(self, shape_data):
        ks = KShape(n_clusters=1)
        ks.fit(shape_data)
        labels = ks.labels_["cluster"].to_list()
        assert all(label == 0 for label in labels)

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        ks = KShape(n_clusters=5)
        with pytest.raises(ValueError, match="Cannot create"):
            ks.fit(df)

    def test_unequal_length_series(self):
        """Test padding path for series with different lengths (line 117)."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 6 + ["C"] * 8,
                "y": ([1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            }
        )
        ks = KShape(n_clusters=2)
        ks.fit(df)
        assert ks.labels_ is not None
        assert ks.labels_.shape[0] == 3

    def test_identical_series_same_cluster(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
                "y": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
            }
        )
        ks = KShape(n_clusters=2)
        ks.fit(df)
        labels = dict(zip(ks.labels_["unique_id"].to_list(), ks.labels_["cluster"].to_list(), strict=False))
        assert labels["A"] == labels["B"]
