"""Tests for permutation feature importance (#59)."""

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402

from polars_ts.importance import permutation_importance  # noqa: E402


def _make_feature_df() -> tuple[pl.DataFrame, LinearRegression]:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)  # noise
    y = 3.0 * x1 + rng.normal(0, 0.1, n)  # y depends on x1 only

    df = pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})
    model = LinearRegression()
    X = df.select("x1", "x2").to_numpy()
    model.fit(X, df["y"].to_numpy())
    return df, model


class TestPermutationImportance:
    def test_basic(self):
        df, model = _make_feature_df()
        result = permutation_importance(df, model, feature_cols=["x1", "x2"])
        assert "feature" in result.columns
        assert "importance_mean" in result.columns
        assert "importance_std" in result.columns
        assert len(result) == 2

    def test_important_feature_ranked_higher(self):
        df, model = _make_feature_df()
        result = permutation_importance(df, model, feature_cols=["x1", "x2"])
        # x1 should be more important than x2
        assert result["feature"][0] == "x1"
        assert result["importance_mean"][0] > result["importance_mean"][1]

    def test_empty_features_raises(self):
        df, model = _make_feature_df()
        with pytest.raises(ValueError, match="non-empty"):
            permutation_importance(df, model, feature_cols=[])

    def test_custom_metric(self):
        df, model = _make_feature_df()

        def rmse_fn(d: pl.DataFrame) -> float:
            import numpy as np

            err = (d["y"] - d["y_hat"]).to_numpy()
            return float(np.sqrt(np.mean(err**2)))

        result = permutation_importance(df, model, feature_cols=["x1", "x2"], metric_fn=rmse_fn)
        assert result["feature"][0] == "x1"


def test_top_level_import():
    import polars_ts

    assert polars_ts.permutation_importance is permutation_importance
