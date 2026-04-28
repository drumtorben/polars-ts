# Clustering & Classification Guide

polars-ts provides 11 clustering algorithms, an automated pipeline selector, 3 evaluation metrics, and 3 classifiers — all working with the same distance infrastructure.

## Available methods

| Method | Function | When to use |
|---|---|---|
| **K-Medoids (PAM)** | `kmedoids` | Known k, any distance metric, interpretable medoids |
| **K-Shape** | `KShape` | Shape-based grouping via cross-correlation centroids |
| **Spectral (KSC)** | `spectral_cluster` | Non-convex clusters, graph Laplacian structure |
| **HDBSCAN** | `hdbscan_cluster` | Unknown k, varying density, noise detection |
| **DBSCAN** | `dbscan_cluster` | Fixed-radius neighbourhood, noise detection |
| **Hierarchical** | `agglomerative_cluster` | Dendrogram visualization, flexible linkage |
| **K-Means DBA** | `kmeans_dba` | DTW-aware centroids |
| **CLARA** | `clara` | Scalable k-medoids via sampling |
| **CLARANS** | `clarans` | Randomized k-medoids neighbourhood search |
| **U-Shapelets** | `shapelet_cluster` | Interpretable sub-sequence patterns |
| **Auto-cluster** | `auto_cluster` | Sweep methods × distances × k, pick the best |

## Quick start

### K-Medoids

```python
import polars_ts as pts

labels = pts.kmedoids(df, k=3, method="dtw")
# → DataFrame[unique_id, cluster]
```

### K-Shape

```python
kshape = pts.KShape(n_clusters=3)
kshape.fit(df)
print(kshape.labels_)
```

### HDBSCAN (automatic k, noise detection)

```python
labels = pts.hdbscan_cluster(df, method="dtw", min_cluster_size=3)
# Noise points have cluster = -1
```

### Spectral clustering

```python
labels = pts.spectral_cluster(df, k=3, method="sbd", sigma=1.0)
```

### Auto-cluster (automated selection)

```python
result = pts.auto_cluster(
    df,
    methods=["kmedoids", "spectral", "kshape"],
    distances=["sbd", "dtw"],
    k_range=range(2, 6),
    metric="silhouette",
)
print(f"Best: {result.best_method}, k={result.best_k}, score={result.best_score:.4f}")
print(result.results_table)  # Full grid search results
```

## Evaluation metrics

```python
from polars_ts import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sil = silhouette_score(df, labels, method="dtw")
dbi = davies_bouldin_score(df, labels, method="dtw")
chi = calinski_harabasz_score(df, labels, method="dtw")
```

| Metric | Ideal value | Interpretation |
|---|---|---|
| **Silhouette** | Close to 1 | Cohesive within, separated between |
| **Davies-Bouldin** | Close to 0 | Low intra-cluster scatter vs inter-cluster distance |
| **Calinski-Harabasz** | Higher is better | Between-cluster to within-cluster dispersion ratio |

## Classification

```python
# k-NN classification
preds = pts.knn_classify(train_df, test_df, k=3, method="dtw", label_col="label")

# OOP API
knn = pts.TimeSeriesKNNClassifier(k=3, metric="dtw")
knn.fit(train_df, label_col="label")
preds = knn.predict(test_df)

# K-Shape classifier
clf = pts.KShapeClassifier(n_centroids_per_class=1)
clf.fit(train_df, label_col="label")
preds = clf.predict(test_df)
```

## Feature extraction for clustering

### ROCKET / MiniRocket

Random convolutional kernel features for downstream clustering.

```python
features = pts.rocket_features(df, n_kernels=10000)
features = pts.minirocket_features(df, n_kernels=10000)
```

### Foundation model embeddings

```python
embeddings = pts.to_chronos_embeddings(df, model_name="amazon/chronos-t5-small")
embeddings = pts.to_moment_embeddings(df, model_name="AutonLab/MOMENT-1-large")
```

## Using distance matrices with scipy/sklearn

polars-ts distance output can be converted to scipy condensed vectors or sklearn precomputed matrices. See the [distance metrics guide](distance-metrics.md) for the full conversion workflow.

```python
from polars_ts import compute_pairwise_dtw
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

distances = compute_pairwise_dtw(df, df)

# Convert to condensed vector
ids = sorted(set(distances["id_1"].to_list()) | set(distances["id_2"].to_list()))
id_to_idx = {name: i for i, name in enumerate(ids)}
n = len(ids)
condensed = np.zeros(n * (n - 1) // 2)
for row in distances.iter_rows(named=True):
    i, j = id_to_idx[row["id_1"]], id_to_idx[row["id_2"]]
    if i > j:
        i, j = j, i
    idx = n * i - i * (i + 1) // 2 + (j - i - 1)
    condensed[idx] = row["dtw"]

# Hierarchical clustering
Z = linkage(condensed, method="average")
labels = fcluster(Z, t=3, criterion="maxclust")
```

## Further reading

- **Notebook 07**: [Time series similarity & clustering](https://github.com/drumtorben/polars-ts/blob/main/notebooks/07_time_series_similarity_clustering.ipynb)
- Paparrizos & Gravano (2015). *k-Shape: Efficient and Accurate Clustering of Time Series*. SIGMOD.
- Campello et al. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.
- von Luxburg (2007). *A Tutorial on Spectral Clustering*. Statistics and Computing.
