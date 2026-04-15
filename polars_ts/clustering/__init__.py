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
    raise AttributeError(f"module 'polars_ts.clustering' has no attribute {name!r}")


__all__ = ["kmedoids", "TimeSeriesKMedoids", "KShape"]
