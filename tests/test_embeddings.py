"""Tests for foundation model embedding adapters.

These tests use lightweight mocks to avoid downloading large models
during CI. Integration tests requiring real models are marked with
``pytest.mark.slow``.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from polars_ts.adapters.embeddings import _arrays_to_result, _extract_series


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "ds": [date(2024, 1, i + 1) for i in range(10)] * 3,
            "y": [float(i) for i in range(10)] + [float(10 - i) for i in range(10)] + [float(i % 3) for i in range(10)],
        }
    )


# ── Helper tests ─────────────────────────────────────────────────────────


class TestExtractSeries:
    def test_ids_and_count(self):
        df = _make_df()
        ids, arrays = _extract_series(df, "y", "unique_id", "ds")
        assert sorted(ids) == ["A", "B", "C"]
        assert len(arrays) == 3

    def test_array_dtype(self):
        df = _make_df()
        _, arrays = _extract_series(df, "y", "unique_id", "ds")
        assert arrays[0].dtype == np.float32

    def test_variable_length(self):
        df = pl.DataFrame(
            {
                "unique_id": ["A"] * 5 + ["B"] * 10,
                "ds": [date(2024, 1, i + 1) for i in range(5)] + [date(2024, 1, i + 1) for i in range(10)],
                "y": [1.0] * 5 + [2.0] * 10,
            }
        )
        ids, arrays = _extract_series(df, "y", "unique_id", "ds")
        assert len(arrays[0]) == 5
        assert len(arrays[1]) == 10

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 3, "t": [1, 2, 3], "val": [1.0, 2.0, 3.0]})
        ids, arrays = _extract_series(df, "val", "sid", "t")
        assert ids == ["X"]
        assert len(arrays[0]) == 3


class TestArraysToResult:
    def test_output_shape(self):
        ids = ["A", "B"]
        emb = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _arrays_to_result(ids, emb, "unique_id", "emb")
        assert result.shape == (2, 4)  # id + 3 dims
        assert result.columns == ["unique_id", "emb_0", "emb_1", "emb_2"]

    def test_values_preserved(self):
        ids = ["X"]
        emb = np.array([[0.5, 1.5]])
        result = _arrays_to_result(ids, emb, "sid", "feat")
        assert result["sid"].to_list() == ["X"]
        assert result["feat_0"].to_list() == [0.5]
        assert result["feat_1"].to_list() == [1.5]


# ── Chronos adapter tests (mocked) ──────────────────────────────────────


class TestChronosEmbeddings:
    def test_import_error_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="torch"):
                # Force reimport
                import importlib

                from polars_ts.adapters import embeddings

                importlib.reload(embeddings)
                embeddings.to_chronos_embeddings(_make_df())

    def test_mocked_chronos(self):
        torch = pytest.importorskip("torch")

        hidden_dim = 8
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(3, 5, hidden_dim)

        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=None)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.__call__ = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.ones(3, 5, dtype=torch.long)}

        with (
            patch("polars_ts.adapters.embeddings.AutoModel") as MockAutoModel,
            patch("polars_ts.adapters.embeddings.AutoTokenizer") as MockAutoTokenizer,
        ):
            MockAutoModel.from_pretrained.return_value = mock_model
            MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from polars_ts.adapters.embeddings import to_chronos_embeddings

            result = to_chronos_embeddings(_make_df(), model_name="fake/model")

        assert result.shape == (3, 1 + hidden_dim)
        assert result.columns[0] == "unique_id"
        assert all(c.startswith("emb_") for c in result.columns[1:])
        assert sorted(result["unique_id"].to_list()) == ["A", "B", "C"]

    def test_mocked_chronos_batch_size(self):
        torch = pytest.importorskip("torch")

        hidden_dim = 4

        def make_output(n):
            out = MagicMock()
            out.last_hidden_state = torch.randn(n, 3, hidden_dim)
            return out

        call_count = {"n": 0}

        def model_call(**kwargs):
            input_ids = kwargs.get("input_ids", list(kwargs.values())[0])
            n = input_ids.shape[0]
            call_count["n"] += 1
            return make_output(n)

        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=None)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.side_effect = model_call

        mock_tokenizer = MagicMock()

        def tokenize(tensors, **_kw):
            n = len(tensors)
            return {"input_ids": torch.ones(n, 3, dtype=torch.long)}

        mock_tokenizer.side_effect = tokenize

        with (
            patch("polars_ts.adapters.embeddings.AutoModel") as MockAutoModel,
            patch("polars_ts.adapters.embeddings.AutoTokenizer") as MockAutoTokenizer,
        ):
            MockAutoModel.from_pretrained.return_value = mock_model
            MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer

            from polars_ts.adapters.embeddings import to_chronos_embeddings

            result = to_chronos_embeddings(_make_df(), model_name="fake/model", batch_size=2)

        assert result.shape[0] == 3
        assert call_count["n"] == 2  # 2 + 1 batches


# ── MOMENT adapter tests (mocked) ───────────────────────────────────────


class TestMomentEmbeddings:
    def test_mocked_moment(self):
        torch = pytest.importorskip("torch")

        emb_dim = 16
        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(3, emb_dim)

        mock_model = MagicMock()
        mock_model.init = MagicMock(return_value=None)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.__call__ = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_output

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"momentfm": MagicMock(MOMENTPipeline=mock_pipeline_cls)}):
            import importlib

            from polars_ts.adapters import embeddings

            importlib.reload(embeddings)
            result = embeddings.to_moment_embeddings(_make_df(), model_name="fake/moment")

        assert result.shape == (3, 1 + emb_dim)
        assert result.columns[0] == "unique_id"
        assert sorted(result["unique_id"].to_list()) == ["A", "B", "C"]


# ── Integration: top-level imports ───────────────────────────────────────


def test_chronos_importable_from_polars_ts():
    from polars_ts import to_chronos_embeddings

    assert callable(to_chronos_embeddings)


def test_moment_importable_from_polars_ts():
    from polars_ts import to_moment_embeddings

    assert callable(to_moment_embeddings)


# ── Additional coverage tests ────────────────────────────────────────────


class TestExtractSeriesNoDsColumn:
    """_extract_series falls back to sorting by id_col only when time_col is absent."""

    def test_no_time_column(self):
        df = pl.DataFrame({"unique_id": ["A", "A", "B"], "y": [1.0, 2.0, 3.0]})
        ids, arrays = _extract_series(df, "y", "unique_id", "ds")
        assert sorted(ids) == ["A", "B"]
        assert len(arrays[0]) == 2
        assert len(arrays[1]) == 1


class TestChronosImportErrorTransformers:
    def test_import_error_transformers(self):
        torch = pytest.importorskip("torch")
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                import importlib

                from polars_ts.adapters import embeddings

                importlib.reload(embeddings)
                embeddings.to_chronos_embeddings(_make_df())


class TestMomentImportErrors:
    def test_import_error_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="torch"):
                import importlib

                from polars_ts.adapters import embeddings

                importlib.reload(embeddings)
                embeddings.to_moment_embeddings(_make_df())

    def test_import_error_momentfm(self):
        torch = pytest.importorskip("torch")
        with patch.dict("sys.modules", {"momentfm": None}):
            with pytest.raises(ImportError, match="momentfm"):
                import importlib

                from polars_ts.adapters import embeddings

                importlib.reload(embeddings)
                embeddings.to_moment_embeddings(_make_df())
