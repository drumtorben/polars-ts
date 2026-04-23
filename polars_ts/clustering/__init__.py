from typing import Any


def __getattr__(name: str) -> Any:
    if name == "kmedoids":
        from polars_ts.clustering.kmedoids import kmedoids

        return kmedoids
    if name == "TimeSeriesKMedoids":
        from polars_ts.clustering.kmedoids import TimeSeriesKMedoids

        return TimeSeriesKMedoids
    if name == "KShape":
        from polars_ts.clustering.kshape import KShape

        return KShape
    if name == "silhouette_score":
        from polars_ts.clustering.evaluation import silhouette_score

        return silhouette_score
    if name == "silhouette_samples":
        from polars_ts.clustering.evaluation import silhouette_samples

        return silhouette_samples
    if name == "davies_bouldin_score":
        from polars_ts.clustering.evaluation import davies_bouldin_score

        return davies_bouldin_score
    if name == "calinski_harabasz_score":
        from polars_ts.clustering.evaluation import calinski_harabasz_score

        return calinski_harabasz_score
    if name == "hdbscan_cluster":
        from polars_ts.clustering.density import hdbscan_cluster

        return hdbscan_cluster
    if name == "dbscan_cluster":
        from polars_ts.clustering.density import dbscan_cluster

        return dbscan_cluster
    raise AttributeError(f"module 'polars_ts.clustering' has no attribute {name!r}")


__all__ = [
    "kmedoids",
    "TimeSeriesKMedoids",
    "KShape",
    "silhouette_score",
    "silhouette_samples",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "hdbscan_cluster",
    "dbscan_cluster",
]
