"""Tests for Isolation Forest adapter (#63)."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")

from polars_ts.anomaly_forest import isolation_forest_detect  # noqa: E402


def _make_df_with_anomaly() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    values = rng.normal(10, 1, n)
    values[25] = 100.0  # obvious anomaly
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "y": values.tolist(),
            "feat1": values.tolist(),
            "feat2": rng.normal(0, 1, n).tolist(),
        }
    )


class TestIsolationForestDetect:
    def test_basic(self):
        df = _make_df_with_anomaly()
        result = isolation_forest_detect(df, feature_cols=["feat1", "feat2"])
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_detects_outlier(self):
        df = _make_df_with_anomaly()
        result = isolation_forest_detect(df, feature_cols=["feat1"], contamination=0.05)
        anomalies = result.filter(pl.col("is_anomaly"))
        assert len(anomalies) >= 1

    def test_per_series(self):
        df = _make_df_with_anomaly()
        df2 = _make_df_with_anomaly().with_columns(pl.lit("B").alias("unique_id"))
        combined = pl.concat([df, df2])
        result = isolation_forest_detect(combined, feature_cols=["feat1", "feat2"], global_model=False)
        assert len(result) == 100

    def test_global_model(self):
        df = _make_df_with_anomaly()
        result = isolation_forest_detect(df, feature_cols=["feat1", "feat2"], global_model=True)
        assert len(result) == 50

    def test_empty_features_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            isolation_forest_detect(_make_df_with_anomaly(), feature_cols=[])


def test_top_level_import():
    import polars_ts

    assert polars_ts.isolation_forest_detect is isolation_forest_detect
