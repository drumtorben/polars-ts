"""Tests for GlobalForecaster (Ch 10)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402

from polars_ts.global_model import GlobalForecaster  # noqa: E402


def _make_ts(n: int = 30, n_series: int = 2) -> pl.DataFrame:
    rows: list[dict] = []
    base = datetime(2024, 1, 1)
    for s in range(n_series):
        sid = chr(65 + s)
        for i in range(n):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(hours=i),
                    "y": float(i * (s + 1)) + 10.0,
                }
            )
    return pl.DataFrame(rows)


def _make_ts_with_static(n: int = 30) -> pl.DataFrame:
    rows: list[dict] = []
    base = datetime(2024, 1, 1)
    for sid, stype in [("A", "retail"), ("B", "warehouse")]:
        scale = 1.0 if stype == "retail" else 3.0
        for i in range(n):
            rows.append(
                {
                    "unique_id": sid,
                    "ds": base + timedelta(hours=i),
                    "y": float(i) * scale + 10.0,
                    "store_type": stype,
                }
            )
    return pl.DataFrame(rows)


class TestGlobalForecaster:
    def test_fit_predict_basic(self):
        df = _make_ts()
        gf = GlobalForecaster(LinearRegression(), lags=[1, 2])
        gf.fit(df)
        result = gf.predict(df, h=3)

        assert result.columns == ["unique_id", "ds", "y_hat"]
        assert len(result) == 6

    def test_ordinal_encoding(self):
        df = _make_ts()
        gf = GlobalForecaster(Ridge(), lags=[1, 2], encode_id="ordinal")
        gf.fit(df)
        result = gf.predict(df, h=2)

        assert len(result) == 4
        assert "__id_encoded" not in result.columns  # internal only

    def test_onehot_encoding(self):
        df = _make_ts()
        gf = GlobalForecaster(Ridge(), lags=[1, 2], encode_id="onehot")
        gf.fit(df)
        result = gf.predict(df, h=2)

        assert len(result) == 4

    def test_static_features(self):
        df = _make_ts_with_static()
        gf = GlobalForecaster(Ridge(), lags=[1, 2], static_features=["store_type"])
        gf.fit(df)
        result = gf.predict(df, h=2)

        assert len(result) == 4

    def test_unseen_series_ordinal(self):
        """Unseen series gets -1 encoding but doesn't crash."""
        df = _make_ts()
        gf = GlobalForecaster(LinearRegression(), lags=[1, 2], encode_id="ordinal")
        gf.fit(df)

        # Create a new series C not seen at fit time
        base = datetime(2024, 1, 1)
        new_df = pl.DataFrame(
            {
                "unique_id": ["C"] * 10,
                "ds": [base + timedelta(hours=i) for i in range(10)],
                "y": [float(i) for i in range(10)],
            }
        )
        result = gf.predict(new_df, h=2)
        assert len(result) == 2

    def test_with_log_transform(self):
        df = _make_ts()
        gf = GlobalForecaster(LinearRegression(), lags=[1, 2], target_transform="log")
        gf.fit(df)
        result = gf.predict(df, h=1)

        for v in result["y_hat"].to_list():
            assert v > 0

    def test_with_difference_transform(self):
        df = _make_ts(n=30, n_series=1)
        gf = GlobalForecaster(LinearRegression(), lags=[1], target_transform="difference")
        gf.fit(df)
        result = gf.predict(df, h=1)

        assert result["y_hat"][0] == pytest.approx(40.0, abs=2.0)

    def test_predict_before_fit(self):
        gf = GlobalForecaster(LinearRegression(), lags=[1])
        with pytest.raises(RuntimeError, match="fit"):
            gf.predict(_make_ts(), h=1)

    def test_no_features_raises(self):
        with pytest.raises(ValueError, match="feature source"):
            GlobalForecaster(LinearRegression())

    def test_invalid_encode_id(self):
        with pytest.raises(ValueError, match="encode_id"):
            GlobalForecaster(LinearRegression(), lags=[1], encode_id="invalid")

    def test_all_features_combined(self):
        df = _make_ts()
        gf = GlobalForecaster(
            Ridge(),
            lags=[1, 2],
            rolling_windows=[3],
            calendar=["hour"],
            encode_id="ordinal",
        )
        gf.fit(df)
        result = gf.predict(df, h=1)

        assert len(result) == 2

    def test_three_series(self):
        df = _make_ts(n_series=3)
        gf = GlobalForecaster(LinearRegression(), lags=[1, 2], encode_id="onehot")
        gf.fit(df)
        result = gf.predict(df, h=2)

        assert len(result) == 6
        assert len(result["unique_id"].unique()) == 3


def test_top_level_import():
    import polars_ts

    assert polars_ts.GlobalForecaster is GlobalForecaster
