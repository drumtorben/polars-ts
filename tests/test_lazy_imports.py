"""Tests for lazy-loaded module imports via polars_ts.__getattr__."""

import ast
import importlib
import sys

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


class TestBayesianLazyImports:
    """T3.2: Bayesian names must be in _LAZY_IMPORTS, not a special-case if-block."""

    BAYESIAN_NAMES = [
        "KalmanFilter",
        "kalman_filter",
        "UnscentedKalmanFilter",
        "EnsembleKalmanFilter",
        "BSTS",
        "bsts_fit",
        "bsts_forecast",
        "GaussianProcessTS",
        "gp_forecast",
        "MCMCForecaster",
        "mcmc_forecast",
    ]

    @pytest.mark.parametrize("name", BAYESIAN_NAMES)
    def test_bayesian_in_lazy_imports(self, name):
        """Every bayesian name should be registered in _LAZY_IMPORTS."""
        assert name in polars_ts._LAZY_IMPORTS, (
            f"{name!r} is not in polars_ts._LAZY_IMPORTS — " f"it may still be in a special-case if-block"
        )

    @pytest.mark.parametrize("name", BAYESIAN_NAMES)
    def test_bayesian_resolves_from_top_level(self, name):
        """Bayesian names must resolve via polars_ts.<name>."""
        obj = getattr(polars_ts, name)
        assert obj is not None

    def test_no_special_case_if_block_in_getattr(self):
        """The __getattr__ in __init__.py must not contain hardcoded name sets."""
        import inspect

        source = inspect.getsource(polars_ts.__getattr__)
        assert "KalmanFilter" not in source, (
            "__getattr__ still contains hardcoded 'KalmanFilter' — " "bayesian names should be in _LAZY_IMPORTS"
        )

    def test_bayesian_in_all(self):
        """All bayesian names should appear in __all__."""
        for name in self.BAYESIAN_NAMES:
            assert name in polars_ts.__all__, f"{name!r} missing from polars_ts.__all__"


class TestStreamingLazyImports:
    """T3.1: streaming/__init__.py must use make_getattr, not eager imports."""

    STREAMING_NAMES = [
        "StreamingETS",
        "StreamingKalmanFilter",
        "StreamingGlobalForecaster",
        "SlidingWindowManager",
    ]

    def test_streaming_uses_make_getattr(self):
        """streaming/__init__.py must define __getattr__ via make_getattr."""
        import polars_ts.streaming as streaming_mod

        source_path = streaming_mod.__file__
        with open(source_path) as f:
            tree = ast.parse(f.read())

        # Should NOT have top-level "from polars_ts.streaming.X import Y" statements
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("polars_ts.streaming."):
                pytest.fail(
                    f"streaming/__init__.py has eager import: 'from {node.module} import ...'. "
                    f"Should use make_getattr pattern instead."
                )

    def test_streaming_has_getattr_from_make_getattr(self):
        """Streaming module should have __getattr__ defined by make_getattr."""
        import polars_ts.streaming as streaming_mod

        assert hasattr(streaming_mod, "__getattr__"), "streaming module missing __getattr__"
        assert callable(streaming_mod.__getattr__)

    @pytest.mark.parametrize("name", STREAMING_NAMES)
    def test_streaming_names_resolve(self, name):
        """All streaming names must still resolve after conversion."""
        import polars_ts.streaming as streaming_mod

        obj = getattr(streaming_mod, name)
        assert obj is not None

    @pytest.mark.parametrize("name", STREAMING_NAMES)
    def test_streaming_names_in_submodule_all(self, name):
        """All streaming names should appear in streaming.__all__."""
        import polars_ts.streaming as streaming_mod

        assert name in streaming_mod.__all__, f"{name!r} missing from streaming.__all__"

    def test_streaming_lazy_import_no_submodule_loaded_at_import_time(self):
        """Importing polars_ts.streaming should not eagerly load submodules."""
        # Remove streaming submodules from sys.modules to test fresh import
        streaming_submodules = [k for k in sys.modules if k.startswith("polars_ts.streaming.")]
        saved = {k: sys.modules.pop(k) for k in streaming_submodules}
        # Also remove streaming itself
        if "polars_ts.streaming" in sys.modules:
            saved["polars_ts.streaming"] = sys.modules.pop("polars_ts.streaming")

        try:
            importlib.import_module("polars_ts.streaming")
            loaded = [k for k in sys.modules if k.startswith("polars_ts.streaming.")]
            assert loaded == [], f"Importing polars_ts.streaming eagerly loaded submodules: {loaded}"
        finally:
            # Restore original modules
            sys.modules.update(saved)


class TestMetricsIntentionalEager:
    """T3.3: metrics/__init__.py uses eager imports intentionally (Polars namespace)."""

    def test_metrics_namespace_registers(self):
        """The Polars .pts namespace must be available after importing polars_ts.metrics."""
        import polars as pl

        import polars_ts.metrics  # noqa: F401 — triggers @register_dataframe_namespace

        df = pl.DataFrame({"y": [1.0, 2.0], "y_hat": [1.1, 2.1]})
        assert hasattr(df, "pts"), "Polars .pts namespace not registered"
        assert hasattr(df.pts, "mae"), "Polars .pts.mae not available"

    def test_metrics_uses_eager_imports_intentionally(self):
        """metrics/__init__.py must use eager imports for Polars namespace registration."""
        import ast

        import polars_ts.metrics as metrics_mod

        source_path = metrics_mod.__file__
        with open(source_path) as f:
            tree = ast.parse(f.read())

        has_register_decorator = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "register_dataframe_namespace":
                has_register_decorator = True
                break
        assert has_register_decorator, (
            "metrics/__init__.py should use @pl.api.register_dataframe_namespace — "
            "it needs eager imports, not make_getattr"
        )
