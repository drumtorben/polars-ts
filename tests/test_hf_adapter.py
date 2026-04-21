"""Tests for HuggingFace adapter (polars_ts.adapters.huggingface). Issue #77."""

from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest

datasets = pytest.importorskip("datasets")

from polars_ts.adapters.huggingface import to_hf_dataset  # noqa: E402


def _make_df(n: int = 10, n_series: int = 1) -> pl.DataFrame:
    ids = []
    ds = []
    ys = []
    for s in range(n_series):
        sid = chr(ord("A") + s)
        base = date(2024, 1, 1)
        for i in range(n):
            ids.append(sid)
            ds.append(base + timedelta(days=i))
            ys.append(float(i + s * 10))
    return pl.DataFrame({"unique_id": ids, "ds": ds, "y": ys})


class TestToHfDataset:
    def test_basic_schema(self):
        result = to_hf_dataset(_make_df())
        assert "id" in result.column_names
        assert "target" in result.column_names
        assert "start" in result.column_names

    def test_multi_series(self):
        result = to_hf_dataset(_make_df(n=10, n_series=2))
        assert len(result) == 2

    def test_values_preserved(self):
        df = _make_df(n=5, n_series=1)
        result = to_hf_dataset(df)
        expected = df.sort("unique_id", "ds")["y"].to_list()
        assert result[0]["target"] == expected

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "series": ["X"] * 5,
                "timestamp": [date(2024, 1, i + 1) for i in range(5)],
                "value": [float(i) for i in range(5)],
            }
        )
        result = to_hf_dataset(df, target_col="value", id_col="series", time_col="timestamp")
        assert result[0]["id"] == "X"
        assert result[0]["target"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_start_timestamps(self):
        df = _make_df(n=10, n_series=2)
        result = to_hf_dataset(df)
        # Start should match the first timestamp per group
        for row in result:
            gid = row["id"]
            group = df.filter(pl.col("unique_id") == gid).sort("ds")
            expected_start = str(group["ds"][0])
            assert row["start"] == expected_start

    def test_sorted_by_id(self):
        df = _make_df(n=5, n_series=3)
        result = to_hf_dataset(df)
        ids = [row["id"] for row in result]
        assert ids == sorted(ids)


def test_import_error():
    """When 'datasets' is not installed, a helpful ImportError is raised."""
    with patch.dict("sys.modules", {"datasets": None}):
        with pytest.raises(ImportError, match="datasets"):
            # Re-import to trigger the error path
            import importlib

            import polars_ts.adapters.huggingface as hf_mod

            importlib.reload(hf_mod)
            hf_mod.to_hf_dataset(_make_df())
