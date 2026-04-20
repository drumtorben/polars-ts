"""Tests for advanced feature engineering (#53)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from polars_ts.features.advanced import interaction_features, target_encode, time_embeddings


def _make_cat_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 6,
            "cat": ["x", "x", "x", "y", "y", "y"],
            "y": [10.0, 12.0, 11.0, 50.0, 52.0, 48.0],
        }
    )


class TestTargetEncode:
    def test_basic(self):
        result = target_encode(_make_cat_df(), cat_col="cat")
        assert "cat_encoded" in result.columns
        # "x" mean ~11, "y" mean ~50
        x_enc = result.filter(pl.col("cat") == "x")["cat_encoded"][0]
        y_enc = result.filter(pl.col("cat") == "y")["cat_encoded"][0]
        assert x_enc < y_enc

    def test_smoothing(self):
        df = _make_cat_df()
        low_smooth = target_encode(df, cat_col="cat", smoothing=0.01)
        high_smooth = target_encode(df, cat_col="cat", smoothing=100.0)
        # High smoothing pulls everything toward global mean
        x_low = low_smooth.filter(pl.col("cat") == "x")["cat_encoded"][0]
        x_high = high_smooth.filter(pl.col("cat") == "x")["cat_encoded"][0]
        global_mean = df["y"].mean()
        assert abs(x_high - global_mean) < abs(x_low - global_mean)  # type: ignore[operator]


class TestInteractionFeatures:
    def test_multiply(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = interaction_features(df, pairs=[("a", "b")])
        assert "a_x_b" in result.columns
        assert result["a_x_b"].to_list() == [4.0, 10.0, 18.0]

    def test_add(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = interaction_features(df, pairs=[("a", "b")], method="add")
        assert "a_plus_b" in result.columns
        assert result["a_plus_b"].to_list() == [4.0, 6.0]

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            interaction_features(pl.DataFrame({"a": [1.0], "b": [2.0]}), pairs=[("a", "b")], method="bad")


class TestTimeEmbeddings:
    def test_basic(self):
        base = datetime(2024, 1, 1)
        df = pl.DataFrame({"ds": [base + timedelta(hours=i) for i in range(24)]})
        result = time_embeddings(df, components=["hour"])
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert len(result) == 24

    def test_multiple_components(self):
        base = datetime(2024, 1, 1)
        df = pl.DataFrame({"ds": [base + timedelta(hours=i) for i in range(48)]})
        result = time_embeddings(df, components=["hour", "day_of_week"])
        assert "hour_sin" in result.columns
        assert "day_of_week_sin" in result.columns

    def test_unknown_component(self):
        df = pl.DataFrame({"ds": [datetime(2024, 1, 1)]})
        with pytest.raises(ValueError, match="Unknown component"):
            time_embeddings(df, components=["invalid"])


def test_top_level_imports():
    import polars_ts

    assert polars_ts.target_encode is target_encode
    assert polars_ts.interaction_features is interaction_features
    assert polars_ts.time_embeddings is time_embeddings
