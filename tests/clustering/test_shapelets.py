"""Tests for U-Shapelet clustering."""

import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.shapelets import UShapeletClusterer, shapelet_cluster


@pytest.fixture
def shapelet_data():
    """Two distinct patterns: spikes vs flat."""
    rng = np.random.default_rng(0)
    flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    spike = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ids = []
    vals = []
    for name, pattern in [("A", flat), ("B", flat), ("C", spike), ("D", spike)]:
        noise = rng.normal(0, 0.01, len(pattern))
        ids.extend([name] * len(pattern))
        vals.extend([float(v + n) for v, n in zip(pattern, noise, strict=False)])
    return pl.DataFrame({"unique_id": ids, "y": vals})


class TestUShapeletClusterer:
    def test_fit_returns_self(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=2, shapelet_lengths=[5, 10], n_candidates=50)
        result = usc.fit(shapelet_data)
        assert result is usc

    def test_labels_shape(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=2, shapelet_lengths=[5, 10], n_candidates=50)
        usc.fit(shapelet_data)
        assert usc.labels_ is not None
        assert usc.labels_.shape[0] == 4
        assert "unique_id" in usc.labels_.columns
        assert "cluster" in usc.labels_.columns

    def test_shapelets_discovered(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=2, n_shapelets=5, shapelet_lengths=[5, 10], n_candidates=50)
        usc.fit(shapelet_data)
        assert len(usc.shapelets_) == 5
        assert all(isinstance(s, np.ndarray) for s in usc.shapelets_)

    def test_deterministic_with_seed(self, shapelet_data):
        usc1 = UShapeletClusterer(n_clusters=2, shapelet_lengths=[5], n_candidates=30, seed=99)
        usc1.fit(shapelet_data)
        usc2 = UShapeletClusterer(n_clusters=2, shapelet_lengths=[5], n_candidates=30, seed=99)
        usc2.fit(shapelet_data)
        assert usc1.labels_.equals(usc2.labels_)

    def test_similar_series_grouped(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=2, n_shapelets=10, shapelet_lengths=[5, 10], n_candidates=200, seed=42)
        usc.fit(shapelet_data)
        labels = dict(zip(usc.labels_["unique_id"].to_list(), usc.labels_["cluster"].to_list(), strict=False))
        # Flat series A,B should cluster together; spike series C,D together
        assert labels["A"] == labels["B"]
        assert labels["C"] == labels["D"]

    def test_too_many_clusters_raises(self):
        df = pl.DataFrame({"unique_id": ["A"] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
        usc = UShapeletClusterer(n_clusters=5, shapelet_lengths=[3])
        with pytest.raises(ValueError, match="Cannot create"):
            usc.fit(df)

    def test_shapelets_count_capped(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=2, n_shapelets=3, shapelet_lengths=[5], n_candidates=10)
        usc.fit(shapelet_data)
        assert len(usc.shapelets_) == 3

    def test_single_cluster(self, shapelet_data):
        usc = UShapeletClusterer(n_clusters=1, shapelet_lengths=[5], n_candidates=20)
        usc.fit(shapelet_data)
        labels = usc.labels_["cluster"].to_list()
        assert all(c == 0 for c in labels)

    def test_variable_length_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 15 + ["B"] * 25 + ["C"] * 20,
                "y": [1.0] * 15 + [5.0] * 25 + [1.0] * 20,
            }
        )
        usc = UShapeletClusterer(n_clusters=2, shapelet_lengths=[5], n_candidates=30)
        usc.fit(df)
        assert usc.labels_ is not None
        assert usc.labels_.shape[0] == 3


class TestShapeletClusterFunction:
    def test_returns_dataframe(self, shapelet_data):
        result = shapelet_cluster(shapelet_data, k=2, shapelet_lengths=[5, 10], n_candidates=50)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 4
        assert "unique_id" in result.columns
        assert "cluster" in result.columns

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "series": ["X"] * 15 + ["Y"] * 15,
                "timestamp": list(range(15)) * 2,
                "value": [float(i) for i in range(30)],
            }
        )
        result = shapelet_cluster(
            df, k=2, shapelet_lengths=[5], n_candidates=30, target_col="value", id_col="series", time_col="timestamp"
        )
        assert result.columns[0] == "series"
        assert result.shape[0] == 2

    def test_deterministic(self, shapelet_data):
        r1 = shapelet_cluster(shapelet_data, k=2, shapelet_lengths=[5], n_candidates=30, seed=7)
        r2 = shapelet_cluster(shapelet_data, k=2, shapelet_lengths=[5], n_candidates=30, seed=7)
        assert r1.equals(r2)


def test_importable_from_polars_ts():
    from polars_ts import shapelet_cluster as sc

    assert callable(sc)


def test_class_importable_from_polars_ts():
    from polars_ts import UShapeletClusterer as USC

    assert USC is not None
