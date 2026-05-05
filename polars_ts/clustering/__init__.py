from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "kmedoids": ("polars_ts.clustering.kmedoids", "kmedoids"),
    "TimeSeriesKMedoids": ("polars_ts.clustering.kmedoids", "TimeSeriesKMedoids"),
    "KShape": ("polars_ts.clustering.kshape", "KShape"),
    "silhouette_score": ("polars_ts.clustering.evaluation", "silhouette_score"),
    "silhouette_samples": ("polars_ts.clustering.evaluation", "silhouette_samples"),
    "davies_bouldin_score": ("polars_ts.clustering.evaluation", "davies_bouldin_score"),
    "calinski_harabasz_score": ("polars_ts.clustering.evaluation", "calinski_harabasz_score"),
    "hdbscan_cluster": ("polars_ts.clustering.density", "hdbscan_cluster"),
    "dbscan_cluster": ("polars_ts.clustering.density", "dbscan_cluster"),
    "spectral_cluster": ("polars_ts.clustering.spectral", "spectral_cluster"),
    "auto_cluster": ("polars_ts.clustering.auto", "auto_cluster"),
    "AutoClusterResult": ("polars_ts.clustering.auto", "AutoClusterResult"),
    "shapelet_cluster": ("polars_ts.clustering.shapelets", "shapelet_cluster"),
    "UShapeletClusterer": ("polars_ts.clustering.shapelets", "UShapeletClusterer"),
    "clara": ("polars_ts.clustering.scalable", "clara"),
    "clarans": ("polars_ts.clustering.scalable", "clarans"),
    "kmeans_dba": ("polars_ts.clustering.kmeans", "kmeans_dba"),
    "TimeSeriesKMeans": ("polars_ts.clustering.kmeans", "TimeSeriesKMeans"),
    "agglomerative_cluster": ("polars_ts.clustering.hierarchical", "agglomerative_cluster"),
    "ContrastiveClusterer": ("polars_ts.clustering.contrastive", "ContrastiveClusterer"),
    "contrastive_cluster": ("polars_ts.clustering.contrastive", "contrastive_cluster"),
    "DECClusterer": ("polars_ts.clustering.deep_cluster", "DECClusterer"),
    "IDECClusterer": ("polars_ts.clustering.deep_cluster", "IDECClusterer"),
    "dec_cluster": ("polars_ts.clustering.deep_cluster", "dec_cluster"),
    "idec_cluster": ("polars_ts.clustering.deep_cluster", "idec_cluster"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
