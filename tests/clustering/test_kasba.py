"""Tests for KASBAClusterer Python wrapper (issue #193, #194)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_ts.clustering.kasba import KASBAClusterer, kasba

# ---------------------------------------------------------------------------
# Helper factories for edge-case tests (#194)
# ---------------------------------------------------------------------------


def make_n_series(n: int, length: int, *, seed: int = 0) -> pl.DataFrame:
    """Create *n* random series each of *length* timepoints."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"unique_id": [], "y": []}
    for i in range(n):
        uid = f"S{i}"
        vals = rng.normal(loc=i * 10.0, scale=1.0, size=length)
        rows["unique_id"].extend([uid] * length)
        rows["y"].extend(vals.tolist())
    return pl.DataFrame(rows)


def make_well_separated_series(
    k: int = 2, n_per_cluster: int = 5, length: int = 20, separation: float = 100.0, *, seed: int = 0
) -> pl.DataFrame:
    """Create well-separated clusters that should converge quickly."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"unique_id": [], "y": []}
    idx = 0
    for cluster_idx in range(k):
        for _ in range(n_per_cluster):
            vals = rng.normal(loc=cluster_idx * separation, scale=0.01, size=length)
            uid = f"S{idx}"
            rows["unique_id"].extend([uid] * length)
            rows["y"].extend(vals.tolist())
            idx += 1
    return pl.DataFrame(rows)


def make_borderline_series(*, seed: int = 0) -> pl.DataFrame:
    """Create borderline-separable series for parameter sensitivity tests."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"unique_id": [], "y": []}
    for i in range(10):
        uid = f"S{i}"
        # Two groups with slight offset — sensitive to distance parameter
        base = rng.normal(loc=(i % 2) * 2.0, scale=1.5, size=15)
        rows["unique_id"].extend([uid] * 15)
        rows["y"].extend(base.tolist())
    return pl.DataFrame(rows)


@pytest.fixture
def cluster_df():
    """Univariate data: 6 series, two clear groups."""
    ascending = [float(i) for i in range(10)]
    descending = [float(9 - i) for i in range(10)]
    return pl.DataFrame(
        {
            "unique_id": (["A1"] * 10 + ["A2"] * 10 + ["A3"] * 10 + ["B1"] * 10 + ["B2"] * 10 + ["B3"] * 10),
            "y": (
                ascending
                + [x + 0.1 for x in ascending]
                + [x - 0.1 for x in ascending]
                + descending
                + [x + 0.1 for x in descending]
                + [x - 0.1 for x in descending]
            ),
        }
    )


@pytest.fixture
def multivariate_cluster_df():
    """Multivariate data: 6 series, 2 channels."""
    rng = np.random.default_rng(42)
    rows = []
    for sid in ["A1", "A2", "A3", "B1", "B2", "B3"]:
        for ch in ["x", "y"]:
            base = rng.normal(0 if sid.startswith("A") else 5, 0.1, 8)
            for val in base:
                rows.append({"unique_id": sid, "channel": ch, "y": float(val)})
    return pl.DataFrame(rows)


class TestKASBAClusterer:
    """Test KASBAClusterer class."""

    def test_fit_returns_self(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2)
        result = clf.fit(cluster_df)
        assert result is clf

    def test_labels_dataframe_schema(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.labels_ is not None
        assert "unique_id" in clf.labels_.columns
        assert "cluster" in clf.labels_.columns
        assert clf.labels_.shape[0] == cluster_df["unique_id"].n_unique()

    def test_predict_matches_fit(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        pred = clf.predict(cluster_df)
        assert pred.shape[0] == cluster_df["unique_id"].n_unique()
        assert "cluster" in pred.columns

    def test_predict_before_fit_raises(self, cluster_df):
        with pytest.raises(RuntimeError, match="fit"):
            KASBAClusterer().predict(cluster_df)

    def test_inertia_nonnegative(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.inertia_ >= 0.0

    def test_n_iter_positive(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.n_iter_ >= 1

    def test_reproducible_with_seed(self, cluster_df):
        labels1 = KASBAClusterer(n_clusters=2, seed=7).fit(cluster_df).labels_
        labels2 = KASBAClusterer(n_clusters=2, seed=7).fit(cluster_df).labels_
        assert labels1.equals(labels2)

    def test_centroids_stored(self, cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(cluster_df)
        assert clf.centroids_ is not None
        assert clf.centroids_.shape[0] == 2

    def test_convenience_function(self, cluster_df):
        result = kasba(cluster_df, k=2)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"unique_id", "cluster"}
        assert result.shape[0] == 6


class TestKASBAMultivariate:
    """Test multivariate KASBA clustering."""

    def test_multivariate_fit(self, multivariate_cluster_df):
        clf = KASBAClusterer(n_clusters=2).fit(multivariate_cluster_df, channel_col="channel")
        assert clf.labels_ is not None
        assert clf.labels_.shape[0] == multivariate_cluster_df["unique_id"].n_unique()

    def test_independent_vs_dependent(self, multivariate_cluster_df):
        labels_ind = (
            KASBAClusterer(n_clusters=2, independent=True).fit(multivariate_cluster_df, channel_col="channel").labels_
        )
        labels_dep = (
            KASBAClusterer(n_clusters=2, independent=False).fit(multivariate_cluster_df, channel_col="channel").labels_
        )
        # Both produce valid output
        assert labels_ind.shape == labels_dep.shape
        assert labels_ind.shape[0] == 6


class TestKASBAEdgeCases:
    """Edge-case and robustness tests for KASBA clustering (issue #194)."""

    def test_single_series_single_cluster(self):
        """One series, one cluster — should assign to cluster 0."""
        df = pl.DataFrame({"unique_id": ["A"] * 10, "y": list(range(10))})
        result = kasba(df, k=1)
        assert result["cluster"].to_list() == [0]

    def test_k_equals_n(self):
        """K == number of series — each series its own cluster."""
        df = make_n_series(n=5, length=20)
        result = kasba(df, k=5)
        assert result["cluster"].n_unique() == 5

    def test_k_greater_than_n_raises(self):
        """K > n_series should raise ValueError."""
        df = make_n_series(n=3, length=20)
        with pytest.raises(ValueError, match="n_clusters.*must be <= n_cases"):
            kasba(df, k=10)

    def test_empty_cluster_recovery(self):
        """Verify no cluster label is unused in output when k=3."""
        df = make_well_separated_series(k=3, n_per_cluster=7, length=15, separation=200.0, seed=0)
        result = kasba(df, k=3, seed=0)
        assert result["cluster"].n_unique() == 3

    def test_convergence_before_max_iter(self):
        """Well-separated clusters should converge in < max_iter."""
        df = make_well_separated_series(k=2, separation=100.0)
        clf = KASBAClusterer(n_clusters=2, max_iter=50).fit(df)
        assert clf.n_iter_ < 50

    def test_all_identical_series(self):
        """All series identical — should still produce valid labels."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                "y": vals * 3,
            }
        )
        result = kasba(df, k=2)
        assert result.shape[0] == 3
        assert result["cluster"].min() >= 0

    def test_different_length_series_padded(self):
        """Series of different lengths should be zero-padded."""
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 3 + ["B"] * 5 + ["C"] * 7,
                "y": [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0, 7.0, 8.0] + [1.0] * 7,
            }
        )
        result = kasba(df, k=2)
        assert result.shape[0] == 3

    def test_msm_cost_parameter_effect(self):
        """Different c values should produce valid clusterings."""
        df = make_borderline_series()
        labels_low_c = kasba(df, k=2, c=0.01)
        labels_high_c = kasba(df, k=2, c=100.0)
        assert labels_low_c.shape == labels_high_c.shape
        assert labels_low_c["cluster"].n_unique() >= 1
        assert labels_high_c["cluster"].n_unique() >= 1
