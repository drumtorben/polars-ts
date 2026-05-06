"""Tests for native multivariate deep forecasting (#165).

TDD: tests define the expected API for MultivariatePatchTST and
iTransformerForecaster before implementation.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mv_panel(n_series: int = 3, n_obs: int = 80) -> pl.DataFrame:
    """Create a multivariate panel with 3 target columns per series."""
    rng = np.random.default_rng(42)
    base = date(2024, 1, 1)
    rows: list[dict] = []
    for sid in [chr(ord("A") + i) for i in range(n_series)]:
        for t in range(n_obs):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(days=t),
                    "y1": float(10 + 0.5 * t + rng.normal(0, 1)),
                    "y2": float(20 - 0.3 * t + rng.normal(0, 1)),
                    "y3": float(5 + 0.1 * t + rng.normal(0, 0.5)),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# MultivariatePatchTST tests
# ---------------------------------------------------------------------------


class TestMultivariatePatchTST:
    def test_fit_returns_self(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = MultivariatePatchTST(
            h=5,
            input_size=32,
            patch_len=8,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        result = model.fit(df)
        assert result is model
        assert model.is_fitted_ is True

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        model = MultivariatePatchTST(h=5, input_size=32, patch_len=8, target_cols=["y1", "y2"])
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(_make_mv_panel())

    def test_predict_returns_all_targets(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = MultivariatePatchTST(
            h=5,
            input_size=32,
            patch_len=8,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        assert isinstance(result, pl.DataFrame)
        assert "y1_hat" in result.columns
        assert "y2_hat" in result.columns
        assert "y3_hat" in result.columns
        assert len(result) == 5  # 1 series * h

    def test_multi_series(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        n_series = 3
        df = _make_mv_panel(n_series=n_series, n_obs=80)
        model = MultivariatePatchTST(
            h=5,
            input_size=32,
            patch_len=8,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        assert len(result) == n_series * 5

    def test_output_has_future_dates(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = MultivariatePatchTST(
            h=5,
            input_size=32,
            patch_len=8,
            target_cols=["y1", "y2"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        last_date = df.sort("ds")["ds"][-1]
        assert result["ds"].min() > last_date

    def test_predictions_are_finite(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = MultivariatePatchTST(
            h=5,
            input_size=32,
            patch_len=8,
            target_cols=["y1", "y2"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        for col in ["y1_hat", "y2_hat"]:
            assert np.all(np.isfinite(result[col].to_numpy()))

    def test_custom_columns(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import MultivariatePatchTST

        df = pl.DataFrame(
            {
                "sid": ["X"] * 50,
                "t": [date(2024, 1, 1) + timedelta(days=i) for i in range(50)],
                "a": np.random.default_rng(42).normal(0, 1, 50).tolist(),
                "b": np.random.default_rng(43).normal(0, 1, 50).tolist(),
            }
        )
        model = MultivariatePatchTST(
            h=3,
            input_size=16,
            patch_len=8,
            target_cols=["a", "b"],
            max_epochs=1,
            id_col="sid",
            time_col="t",
        )
        model.fit(df)
        result = model.predict(df)
        assert "sid" in result.columns
        assert "t" in result.columns
        assert "a_hat" in result.columns
        assert "b_hat" in result.columns


# ---------------------------------------------------------------------------
# iTransformerForecaster tests
# ---------------------------------------------------------------------------


class TestITransformerForecaster:
    def test_fit_returns_self(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = iTransformerForecaster(
            h=5,
            input_size=30,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        result = model.fit(df)
        assert result is model
        assert model.is_fitted_ is True

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        model = iTransformerForecaster(h=5, input_size=30, target_cols=["y1", "y2"])
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(_make_mv_panel())

    def test_predict_returns_all_targets(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = iTransformerForecaster(
            h=5,
            input_size=30,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        assert "y1_hat" in result.columns
        assert "y2_hat" in result.columns
        assert "y3_hat" in result.columns
        assert len(result) == 5

    def test_multi_series(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        n_series = 2
        df = _make_mv_panel(n_series=n_series, n_obs=80)
        model = iTransformerForecaster(
            h=3,
            input_size=30,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        assert len(result) == n_series * 3

    def test_predictions_are_finite(self):
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = iTransformerForecaster(
            h=5,
            input_size=30,
            target_cols=["y1", "y2"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        for col in ["y1_hat", "y2_hat"]:
            assert np.all(np.isfinite(result[col].to_numpy()))

    def test_variate_tokens(self):
        """ITransformer treats each variate as a token — more variates = more tokens."""
        pytest.importorskip("torch")
        from polars_ts.dl.multivariate import iTransformerForecaster

        df = _make_mv_panel(n_series=1, n_obs=80)
        model = iTransformerForecaster(
            h=5,
            input_size=30,
            target_cols=["y1", "y2", "y3"],
            max_epochs=1,
        )
        model.fit(df)
        result = model.predict(df)
        assert len(result) == 5
        assert len([c for c in result.columns if c.endswith("_hat")]) == 3


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_importable_from_dl(self):
        pytest.importorskip("torch")
        from polars_ts.dl import MultivariatePatchTST, iTransformerForecaster

        assert MultivariatePatchTST is not None
        assert iTransformerForecaster is not None

    def test_importable_from_top_level(self):
        pytest.importorskip("torch")
        from polars_ts import MultivariatePatchTST, iTransformerForecaster

        assert MultivariatePatchTST is not None
        assert iTransformerForecaster is not None
