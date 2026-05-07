"""Validation and benchmark tests for KASBA clustering (issue #195)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars_ts_rs.polars_ts_rs import kasba_fit

from polars_ts.clustering.kasba import kasba

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from aeon.clustering import TimeSeriesKMeans

    HAS_AEON = True
except ImportError:
    HAS_AEON = False

# ---------------------------------------------------------------------------
# Synthetic data generators with known ground-truth labels
# ---------------------------------------------------------------------------


def make_synthetic_clusters(
    k: int = 3,
    n_per_cluster: int = 20,
    length: int = 50,
    noise: float = 0.1,
    *,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[int]]:
    """Create synthetic time series with known cluster membership.

    Each cluster has a distinct base pattern (sinusoid with different
    frequency), and series within a cluster are noisy copies.

    Returns (DataFrame, ground_truth_labels) where labels are ordered
    by sorted unique_id.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"unique_id": [], "y": []}
    labels: dict[str, int] = {}
    t = np.linspace(0, 2 * np.pi, length)

    idx = 0
    for cluster in range(k):
        # Each cluster gets a distinct sinusoidal frequency
        freq = 1.0 + cluster * 2.0
        base = np.sin(freq * t) * (cluster + 1)
        for _ in range(n_per_cluster):
            uid = f"S{idx:04d}"
            vals = base + rng.normal(0, noise, size=length)
            rows["unique_id"].extend([uid] * length)
            rows["y"].extend(vals.tolist())
            labels[uid] = cluster
            idx += 1

    df = pl.DataFrame(rows)
    # Return labels sorted by unique_id to match kasba() output order
    sorted_ids = sorted(labels.keys())
    true_labels = [labels[uid] for uid in sorted_ids]
    return df, true_labels


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestKASBAValidation:
    """Validate KASBA clustering quality against known ground truth."""

    def test_ari_on_synthetic_clusters(self):
        """Known ground-truth clusters — ARI should be > 0.9."""
        df, true_labels = make_synthetic_clusters(k=3, n_per_cluster=20, length=50, noise=0.1)
        result = kasba(df, k=3, seed=42)
        predicted = result.sort("unique_id")["cluster"].to_list()
        ari = adjusted_rand_score(true_labels, predicted)
        assert ari > 0.9, f"ARI {ari:.3f} is below 0.9 threshold"

    def test_nmi_on_synthetic_clusters(self):
        """NMI should be > 0.85 on well-separated synthetic data."""
        df, true_labels = make_synthetic_clusters(k=3, n_per_cluster=20, length=50, noise=0.1)
        result = kasba(df, k=3, seed=42)
        predicted = result.sort("unique_id")["cluster"].to_list()
        nmi = normalized_mutual_info_score(true_labels, predicted)
        assert nmi > 0.85, f"NMI {nmi:.3f} is below 0.85 threshold"

    def test_ari_degrades_with_noise(self):
        """Higher noise should produce lower (but still reasonable) ARI."""
        df_low, labels_low = make_synthetic_clusters(k=3, n_per_cluster=15, length=40, noise=0.05)
        df_high, labels_high = make_synthetic_clusters(k=3, n_per_cluster=15, length=40, noise=0.5)

        result_low = kasba(df_low, k=3, seed=42)
        result_high = kasba(df_high, k=3, seed=42)

        ari_low = adjusted_rand_score(labels_low, result_low.sort("unique_id")["cluster"].to_list())
        ari_high = adjusted_rand_score(labels_high, result_high.sort("unique_id")["cluster"].to_list())

        assert ari_low >= ari_high, f"Low-noise ARI {ari_low:.3f} < high-noise ARI {ari_high:.3f}"
        assert ari_low > 0.8, f"Low-noise ARI {ari_low:.3f} is below 0.8"

    @pytest.mark.skipif(not HAS_AEON, reason="aeon not installed")
    def test_clustering_quality_vs_aeon(self):
        """ARI between polars-ts KASBA and aeon should be > 0.8."""
        df, true_labels = make_synthetic_clusters(k=3, n_per_cluster=15, length=30, noise=0.1)

        # polars-ts KASBA
        result_pts = kasba(df, k=3, seed=42)
        predicted_pts = result_pts.sort("unique_id")["cluster"].to_list()

        # aeon reference
        sorted_ids = sorted(df["unique_id"].unique().to_list())
        data_3d = np.zeros((len(sorted_ids), 1, 30))
        for i, uid in enumerate(sorted_ids):
            vals = df.filter(pl.col("unique_id") == uid)["y"].to_numpy()
            data_3d[i, 0, : len(vals)] = vals

        aeon_km = TimeSeriesKMeans(n_clusters=3, metric="msm", random_state=42)
        aeon_labels = aeon_km.fit_predict(data_3d)

        ari = adjusted_rand_score(aeon_labels, predicted_pts)
        assert ari > 0.8, f"Cross-impl ARI {ari:.3f} is below 0.8 threshold"


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestKASBABenchmark:
    """Benchmark KASBA performance using pytest-benchmark."""

    def test_kasba_fit_100x100(self, benchmark):
        """Benchmark full KASBA fit on 100 series x 100 timepoints."""
        rng = np.random.default_rng(0)
        data = np.zeros((100, 1, 100), dtype=np.float64)
        for i in range(100):
            cluster = i % 5
            data[i, 0, :] = rng.normal(loc=cluster * 10.0, scale=0.5, size=100)

        result = benchmark(
            kasba_fit,
            data,
            n_clusters=5,
            c=1.0,
            independent=True,
            ba_subset_size=0.5,
            initial_step_size=0.05,
            decay_rate=0.1,
            n_iters=10,
            random_seed=42,
        )
        labels, centers, inertia, n_iter = result
        assert len(labels) == 100
        assert centers.shape[0] == 5

    def test_kasba_fit_50x200(self, benchmark):
        """Benchmark KASBA fit on 50 series x 200 timepoints (longer series)."""
        rng = np.random.default_rng(0)
        data = np.zeros((50, 1, 200), dtype=np.float64)
        for i in range(50):
            cluster = i % 3
            data[i, 0, :] = rng.normal(loc=cluster * 20.0, scale=1.0, size=200)

        result = benchmark(
            kasba_fit,
            data,
            n_clusters=3,
            c=1.0,
            independent=True,
            ba_subset_size=0.5,
            initial_step_size=0.05,
            decay_rate=0.1,
            n_iters=10,
            random_seed=42,
        )
        labels, centers, inertia, n_iter = result
        assert len(labels) == 50
        assert centers.shape[0] == 3
