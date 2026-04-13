# Clustering Time Series with polars-ts Distance Matrices

This guide shows how to use pairwise distance matrices computed by polars-ts as
input to standard clustering algorithms from scipy and scikit-learn.

## Overview

polars-ts computes pairwise distances between time series and returns the results
as a Polars DataFrame with three columns:

| Column | Description |
|--------|-------------|
| `id_1` | Identifier of the first series |
| `id_2` | Identifier of the second series |
| *(metric)* | The distance value (column name matches the metric, e.g. `dtw`, `msm`) |

The output contains one row per **unique unordered pair** — that is, only the
upper triangle of the distance matrix, excluding the diagonal. This is exactly
the information needed to build a condensed distance vector for scipy or a full
square matrix for scikit-learn.

---

## 1. Computing a Full Pairwise Distance Matrix

All `compute_pairwise_*` functions accept two DataFrames with columns
`unique_id` (series identifier) and `y` (values). Pass the same DataFrame twice
to get the full pairwise matrix of all series against each other.

```python
import polars as pl
from polars_ts import compute_pairwise_dtw
# Each series is identified by "unique_id"; values live in "y"
df = pl.DataFrame({
    "unique_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    "y":         [1.0, 2, 3, 4, 5,
                  1.0, 2, 3, 4, 6,
                  5.0, 4, 3, 2, 1],
})

distances = compute_pairwise_dtw(df, df)
print(distances)
# shape: (3, 3)
# ┌──────┬──────┬─────┐
# │ id_1 │ id_2 │ dtw │
# │ ---  │ ---  │ --- │
# │ str  │ str  │ f64 │
# ╞══════╪══════╪═════╡
# │ A    │ B    │ 1.0 │
# │ A    │ C    │ 8.0 │
# │ B    │ C    │ 9.0 │
# └──────┴──────┴─────┘
```

Other available distance functions and their key parameters:

| Function | Distance column | Key parameters |
|----------|----------------|----------------|
| `compute_pairwise_dtw` | `dtw` | `method` (`"sakoe_chiba"`, `"itakura"`, `"fast"`), `param` |
| `compute_pairwise_ddtw` | `ddtw` | — |
| `compute_pairwise_wdtw` | `wdtw` | `g` (weight penalty) |
| `compute_pairwise_msm` | `msm` | `c` (cost) |
| `compute_pairwise_erp` | `erp` | `g` (gap penalty) |
| `compute_pairwise_lcss` | `lcss` | `epsilon` (matching threshold) |
| `compute_pairwise_twe` | `twe` | `nu`, `lambda_` |
| `compute_pairwise_dtw_multi` | `dtw_multi` | `metric` (`"manhattan"`, `"euclidean"`) |
| `compute_pairwise_msm_multi` | `msm_multi` | `c` |

---

## 2. Converting to a scipy Condensed Distance Vector

scipy's hierarchical clustering functions expect a **condensed distance vector**
— a 1-D array containing the upper triangle of the distance matrix in
row-major order. polars-ts already returns exactly these upper-triangle pairs,
but the row order may not match what scipy expects. The helper below
re-indexes the pairs into the correct order.

```python
import numpy as np
from scipy.spatial.distance import squareform

def to_condensed(distances):
    # Convert a polars-ts pairwise result to a scipy condensed distance vector.
    # Returns a (condensed_vector, labels) tuple.
    cols = [c for c in distances.columns if c not in ("id_1", "id_2")]
dist_col = cols[0]
    # Collect all unique ids in sorted order
    all_ids = set(distances["id_1"].to_list()) | set(distances["id_2"].to_list())
    ids = sorted(all_ids)
    id_to_idx = {name: i for i, name in enumerate(ids)}
    n = len(ids)
    # Build the condensed vector in scipy's expected order
    condensed = np.zeros(n * (n - 1) // 2)
    for row in distances.iter_rows(named=True):
        i, j = id_to_idx[row["id_1"]], id_to_idx[row["id_2"]]
        if i > j:
            i, j = j, i
        idx = n * i - i * (i + 1) // 2 + (j - i - 1)
        condensed[idx] = row[dist_col]
    return condensed, ids
```

You can also convert to a full square matrix when needed:

```python
condensed, labels = to_condensed(distances)
square_matrix = squareform(condensed)
print(square_matrix)
```

---

## 3. Hierarchical Clustering with scipy

Once you have the condensed vector you can pass it directly to
`scipy.cluster.hierarchy.linkage`.

```python
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

condensed, labels = to_condensed(distances)
# Compute the linkage matrix (Ward's method is not valid for arbitrary
# distance matrices; use "average", "complete", or "single" instead)
Z = linkage(condensed, method="average")
# Cut the dendrogram to get flat clusters
cluster_labels = fcluster(Z, t=2, criterion="maxclust")
for name, cluster in zip(labels, cluster_labels):
    print(f"  {name} -> cluster {cluster}")
```

### Drawing a Dendrogram

```python
fig, ax = plt.subplots(figsize=(8, 4))
dendrogram(Z, labels=labels, ax=ax)
ax.set_ylabel("Distance")
ax.set_title("Hierarchical Clustering Dendrogram (DTW)")
plt.tight_layout()
plt.show()
```

---

## 4. Spectral Clustering with scikit-learn

Spectral clustering operates on an **affinity (similarity) matrix** rather than a
distance matrix. Convert distances to affinities using a Gaussian kernel or
simple inversion.

```python
from sklearn.cluster import SpectralClustering

condensed, labels = to_condensed(distances)
square_matrix = squareform(condensed)
# Convert distances to affinities via a Gaussian kernel.
# sigma controls the width; a common heuristic is the median distance.
sigma = np.median(square_matrix[square_matrix > 0])
affinity_matrix = np.exp(-square_matrix ** 2 / (2 * sigma ** 2))
# Ensure the diagonal is 1 (self-similarity)
np.fill_diagonal(affinity_matrix, 1.0)

sc = SpectralClustering(
    n_clusters=2,
    affinity="precomputed",
    random_state=42,
)
cluster_labels = sc.fit_predict(affinity_matrix)

for name, cluster in zip(labels, cluster_labels):
    print(f"  {name} -> cluster {cluster}")
```

---

## 5. Complete Worked Example

The example below generates synthetic time series, computes DTW distances,
performs both hierarchical and spectral clustering, and prints the results.

```python
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

from polars_ts import compute_pairwise_dtw
# ── Step 1: Create sample time series ──────────────────────────────────
np.random.seed(42)
n_points = 50
# Group 1: upward trends
series = {}
for i in range(4):
    series[f"up_{i}"] = np.linspace(0, 10, n_points) + np.random.normal(0, 0.5, n_points)
# Group 2: downward trends
for i in range(4):
    series[f"down_{i}"] = np.linspace(10, 0, n_points) + np.random.normal(0, 0.5, n_points)
# Build a polars DataFrame in the expected format
df = pl.DataFrame({
    "unique_id": [name for name, vals in series.items() for _ in vals],
    "y": [v for vals in series.values() for v in vals],
})

print(f"Input shape: {df.shape}")
print(f"Series: {sorted(series.keys())}")
# ── Step 2: Compute pairwise DTW distances ─────────────────────────────
distances = compute_pairwise_dtw(df, df)
print(f"\nPairwise distances: {distances.shape[0]} pairs")
print(distances.head())
# ── Step 3: Convert to condensed form ──────────────────────────────────
cols = [c for c in distances.columns if c not in ("id_1", "id_2")]
dist_col = cols[0]
ids = sorted(set(distances["id_1"].to_list()) | set(distances["id_2"].to_list()))
id_to_idx = {name: i for i, name in enumerate(ids)}
n = len(ids)

condensed = np.zeros(n * (n - 1) // 2)
for row in distances.iter_rows(named=True):
    i, j = id_to_idx[row["id_1"]], id_to_idx[row["id_2"]]
    if i > j:
        i, j = j, i
    idx = n * i - i * (i + 1) // 2 + (j - i - 1)
    condensed[idx] = row[dist_col]
# ── Step 4: Hierarchical clustering ───────────────────────────────────
Z = linkage(condensed, method="average")
hier_labels = fcluster(Z, t=2, criterion="maxclust")

print("\n=== Hierarchical Clustering (2 clusters) ===")
for name, cl in zip(ids, hier_labels):
    print(f"  {name:>8s} -> cluster {cl}")
# ── Step 5: Spectral clustering ───────────────────────────────────────
square_matrix = squareform(condensed)
sigma = np.median(square_matrix[square_matrix > 0])
affinity = np.exp(-square_matrix ** 2 / (2 * sigma ** 2))
np.fill_diagonal(affinity, 1.0)

sc = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=42)
spec_labels = sc.fit_predict(affinity)

print("\n=== Spectral Clustering (2 clusters) ===")
for name, cl in zip(ids, spec_labels):
    print(f"  {name:>8s} -> cluster {cl}")
# ── Step 6: Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Dendrogram
dendrogram(Z, labels=ids, ax=axes[0])
axes[0].set_title("Hierarchical Clustering Dendrogram")
axes[0].set_ylabel("DTW Distance")
axes[0].tick_params(axis="x", rotation=45)
# Heatmap of the distance matrix
im = axes[1].imshow(square_matrix, cmap="viridis")
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
axes[1].set_xticklabels(ids, rotation=45, ha="right")
axes[1].set_yticklabels(ids)
axes[1].set_title("DTW Distance Heatmap")
fig.colorbar(im, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.show()
```

### Expected output

The eight series should cleanly separate into two clusters: all `up_*` series in
one cluster and all `down_*` series in the other. Both hierarchical and spectral
clustering should agree on this partition because the within-group DTW distances
(small warping between noisy versions of the same trend) are much smaller than the
between-group distances (comparing an upward trend against a downward trend).

---

## Tips

- **Linkage method**: Ward's method (`method="ward"`) requires Euclidean
  distances and is not appropriate for arbitrary time-series distance metrics.
  Use `"average"`, `"complete"`, or `"single"` instead.
- **Choosing the number of clusters**: Use the dendrogram to visually identify a
  natural cut height, or use the `inconsistent` criterion in `fcluster`.
- **Large datasets**: polars-ts distance functions are implemented in Rust and
  run in parallel. For very large collections, consider using approximate methods
  like `compute_pairwise_dtw(df, df, method="fast", param=5.0)` (FastDTW).
- **Alternative metrics**: Different distance metrics capture different notions
  of similarity. MSM is better for amplitude-sensitive comparisons, LCSS is
  robust to outliers, and WDTW penalizes large temporal shifts. Experiment with
  multiple metrics on your data.
