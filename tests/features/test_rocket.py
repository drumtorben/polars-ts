"""Tests for ROCKET and MiniRocket feature extraction."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from polars_ts.features.rocket import (
    minirocket_features,
    rocket_features,
)


def _make_df() -> pl.DataFrame:
    """Three short time series with distinct patterns."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "ds": [date(2024, 1, i + 1) for i in range(20)] * 3,
            "y": (
                [float(i) for i in range(20)] + [float(20 - i) for i in range(20)] + [float(i % 5) for i in range(20)]
            ),
        }
    )


# ── ROCKET tests ────────────────────────────────────────────────────────


class TestRocket:
    def test_output_shape(self):
        df = _make_df()
        result = rocket_features(df, n_kernels=10)
        assert result.shape == (3, 1 + 2 * 10)  # id + 2 features per kernel

    def test_column_names(self):
        df = _make_df()
        result = rocket_features(df, n_kernels=5)
        assert result.columns[0] == "unique_id"
        for i in range(10):
            assert f"rocket_{i}" in result.columns

    def test_deterministic_with_seed(self):
        df = _make_df()
        r1 = rocket_features(df, n_kernels=10, seed=42)
        r2 = rocket_features(df, n_kernels=10, seed=42)
        assert r1.equals(r2)

    def test_different_seeds_differ(self):
        df = _make_df()
        r1 = rocket_features(df, n_kernels=10, seed=1)
        r2 = rocket_features(df, n_kernels=10, seed=2)
        vals1 = r1.drop("unique_id").to_numpy()
        vals2 = r2.drop("unique_id").to_numpy()
        assert not np.allclose(vals1, vals2)

    def test_ppv_in_range(self):
        """PPV features (even columns) must be in [0, 1]."""
        df = _make_df()
        result = rocket_features(df, n_kernels=20)
        for i in range(0, 40, 2):
            col = result[f"rocket_{i}"].to_numpy()
            assert np.all(col >= 0.0) and np.all(col <= 1.0)

    def test_preserves_series_ids(self):
        df = _make_df()
        result = rocket_features(df, n_kernels=5)
        assert sorted(result["unique_id"].to_list()) == ["A", "B", "C"]

    def test_variable_length_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 30 + ["B"] * 15,
                "ds": ([date(2024, 1, i + 1) for i in range(30)] + [date(2024, 1, i + 1) for i in range(15)]),
                "y": [1.0] * 30 + [2.0] * 15,
            }
        )
        result = rocket_features(df, n_kernels=5)
        assert result.shape[0] == 2

    def test_n_kernels_validation(self):
        df = _make_df()
        with pytest.raises(ValueError, match="n_kernels"):
            rocket_features(df, n_kernels=0)

    def test_distinct_series_produce_different_features(self):
        df = _make_df()
        result = rocket_features(df, n_kernels=50)
        a = result.filter(pl.col("unique_id") == "A").drop("unique_id").to_numpy()
        b = result.filter(pl.col("unique_id") == "B").drop("unique_id").to_numpy()
        assert not np.allclose(a, b)


# ── MiniRocket tests ───────────────────────────────────────────────────


class TestMiniRocket:
    def test_output_shape(self):
        df = _make_df()
        result = minirocket_features(df, n_kernels=84)
        assert result.shape[0] == 3
        assert result.shape[1] > 1  # id + features

    def test_column_prefix(self):
        df = _make_df()
        result = minirocket_features(df, n_kernels=10)
        feat_cols = [c for c in result.columns if c != "unique_id"]
        assert all(c.startswith("minirocket_") for c in feat_cols)

    def test_deterministic_with_seed(self):
        df = _make_df()
        r1 = minirocket_features(df, n_kernels=84, seed=0)
        r2 = minirocket_features(df, n_kernels=84, seed=0)
        assert r1.equals(r2)

    def test_ppv_in_range(self):
        """All MiniRocket features (PPV) must be in [0, 1]."""
        df = _make_df()
        result = minirocket_features(df, n_kernels=84)
        feat = result.drop("unique_id").to_numpy()
        assert np.all(feat >= 0.0) and np.all(feat <= 1.0)

    def test_preserves_series_ids(self):
        df = _make_df()
        result = minirocket_features(df, n_kernels=10)
        assert sorted(result["unique_id"].to_list()) == ["A", "B", "C"]

    def test_n_kernels_validation(self):
        df = _make_df()
        with pytest.raises(ValueError, match="n_kernels"):
            minirocket_features(df, n_kernels=0)

    def test_distinct_series_produce_different_features(self):
        df = _make_df()
        result = minirocket_features(df, n_kernels=84)
        a = result.filter(pl.col("unique_id") == "A").drop("unique_id").to_numpy()
        c = result.filter(pl.col("unique_id") == "C").drop("unique_id").to_numpy()
        assert not np.allclose(a, c)

    def test_variable_length_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 30 + ["B"] * 15,
                "ds": ([date(2024, 1, i + 1) for i in range(30)] + [date(2024, 1, i + 1) for i in range(15)]),
                "y": [1.0] * 30 + [2.0] * 15,
            }
        )
        result = minirocket_features(df, n_kernels=84)
        assert result.shape[0] == 2


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_series(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 20,
                "ds": [date(2024, 1, i + 1) for i in range(20)],
                "y": [float(i) for i in range(20)],
            }
        )
        r = rocket_features(df, n_kernels=5)
        assert r.shape == (1, 11)
        m = minirocket_features(df, n_kernels=10)
        assert m.shape[0] == 1

    def test_custom_column_names(self):
        df = pl.DataFrame(
            {
                "series": ["X"] * 15 + ["Y"] * 15,
                "timestamp": [date(2024, 1, i + 1) for i in range(15)] * 2,
                "value": [float(i) for i in range(30)],
            }
        )
        r = rocket_features(df, n_kernels=5, target_col="value", id_col="series", time_col="timestamp")
        assert r.columns[0] == "series"
        assert r.shape == (2, 11)

        m = minirocket_features(df, n_kernels=10, target_col="value", id_col="series", time_col="timestamp")
        assert m.columns[0] == "series"
        assert m.shape[0] == 2


# ── Integration: top-level import ──────────────────────────────────────


def test_rocket_importable_from_polars_ts():
    from polars_ts import rocket_features as rf

    assert callable(rf)


def test_minirocket_importable_from_polars_ts():
    from polars_ts import minirocket_features as mf

    assert callable(mf)
