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


def test_kshape_fit_predict_consistent(shape_data):
    """Fitting twice on the same data should give consistent cluster count."""
    ks1 = KShape(n_clusters=2, max_iter=50)
    ks1.fit(shape_data)
    ks2 = KShape(n_clusters=2, max_iter=50)
    ks2.fit(shape_data)
    assert ks1.labels_["cluster"].n_unique() == ks2.labels_["cluster"].n_unique()


def test_kshape_centroids_shape(shape_data):
    """Centroids should have the right length."""
    ks = KShape(n_clusters=2)
    ks.fit(shape_data)
    # Each series has 8 points, so centroids should also have length 8
    for centroid in ks.centroids_:
        assert len(centroid) == 8


def test_kshape_constant_series():
    """Constant series should not crash."""
    df = pl.DataFrame(
        {
            "unique_id": ["A"] * 4 + ["B"] * 4,
            "y": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    ks = KShape(n_clusters=2)
    ks.fit(df)
    assert ks.labels_ is not None
