def __getattr__(name: str):
    if name == "kmedoids":
        from polars_ts.clustering.kmedoids import kmedoids

        return kmedoids
    raise AttributeError(f"module 'polars_ts.clustering' has no attribute {name!r}")


__all__ = ["kmedoids"]
