def __getattr__(name: str):
    if name == "knn_classify":
        from polars_ts.classification.knn import knn_classify

        return knn_classify
    raise AttributeError(f"module 'polars_ts.classification' has no attribute {name!r}")


__all__ = ["knn_classify"]
