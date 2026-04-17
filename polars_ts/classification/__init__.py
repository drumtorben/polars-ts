from typing import Any


def __getattr__(name: str) -> Any:
    if name == "knn_classify":
        from polars_ts.classification.knn import knn_classify

        return knn_classify
    if name == "TimeSeriesKNNClassifier":
        from polars_ts.classification.knn import TimeSeriesKNNClassifier

        return TimeSeriesKNNClassifier
    if name == "KShapeClassifier":
        from polars_ts.classification.kshape_classifier import KShapeClassifier

        return KShapeClassifier
    raise AttributeError(f"module 'polars_ts.classification' has no attribute {name!r}")


__all__ = ["knn_classify", "TimeSeriesKNNClassifier", "KShapeClassifier"]
