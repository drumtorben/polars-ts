from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "knn_classify": ("polars_ts.classification.knn", "knn_classify"),
    "TimeSeriesKNNClassifier": ("polars_ts.classification.knn", "TimeSeriesKNNClassifier"),
    "KShapeClassifier": ("polars_ts.classification.kshape_classifier", "KShapeClassifier"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
