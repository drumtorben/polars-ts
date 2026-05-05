from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "knn_classify": ("polars_ts.classification.knn", "knn_classify"),
    "TimeSeriesKNNClassifier": ("polars_ts.classification.knn", "TimeSeriesKNNClassifier"),
    "KShapeClassifier": ("polars_ts.classification.kshape_classifier", "KShapeClassifier"),
    "RocketClassifier": ("polars_ts.classification.rocket_classifier", "RocketClassifier"),
    "MiniRocketClassifier": ("polars_ts.classification.rocket_classifier", "MiniRocketClassifier"),
    "InceptionTimeClassifier": ("polars_ts.classification.inception_time", "InceptionTimeClassifier"),
    "ResNetClassifier": ("polars_ts.classification.resnet_classifier", "ResNetClassifier"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
