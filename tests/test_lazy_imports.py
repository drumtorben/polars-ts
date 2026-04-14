"""Tests for lazy-loaded module imports via polars_ts.__getattr__."""

import pytest

import polars_ts


class TestLazyImports:
    """Verify that all lazy-loaded attributes are accessible from the top-level package."""

    @pytest.mark.parametrize(
        "name",
        [
            "cusum",
            "seasonal_decomposition",
            "seasonal_decompose_features",
            "TimeSeriesKNNClassifier",
            "KShapeClassifier",
            "TimeSeriesKMedoids",
            "KShape",
        ],
    )
    def test_getattr_resolves(self, name):
        obj = getattr(polars_ts, name)
        assert obj is not None

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            _ = polars_ts.this_does_not_exist  # type: ignore[attr-defined]

    def test_knn_classifier_is_correct_class(self):
        from polars_ts.classification.knn import TimeSeriesKNNClassifier

        assert polars_ts.TimeSeriesKNNClassifier is TimeSeriesKNNClassifier

    def test_kshape_is_correct_class(self):
        from polars_ts.clustering.kshape import KShape

        assert polars_ts.KShape is KShape

    def test_kmedoids_is_correct_class(self):
        from polars_ts.clustering.kmedoids import TimeSeriesKMedoids

        assert polars_ts.TimeSeriesKMedoids is TimeSeriesKMedoids

    def test_kshape_classifier_is_correct_class(self):
        from polars_ts.classification.kshape_classifier import KShapeClassifier

        assert polars_ts.KShapeClassifier is KShapeClassifier

    def test_cusum_is_callable(self):
        assert callable(polars_ts.cusum)
