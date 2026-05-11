"""Tests for dataset loading with integrity verification (#255)."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import polars as pl
import pytest

from polars_ts.datasets import REGISTRY, load_dataset


class TestDatasetRegistry:
    def test_registry_contains_known_datasets(self):
        assert "m4-hourly" in REGISTRY
        assert "air-passengers" in REGISTRY
        assert "m5_y" in REGISTRY

    def test_registry_entries_have_required_keys(self):
        for name, entry in REGISTRY.items():
            assert "url" in entry, f"{name} missing 'url'"
            assert "sha256" in entry, f"{name} missing 'sha256'"
            assert entry["url"].startswith("https://"), f"{name} URL not HTTPS"


class TestLoadDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="unknown-dataset"):
            load_dataset("unknown-dataset")

    def test_hash_mismatch_raises(self, tmp_path):
        # Write a file at the expected cache location so it's "already downloaded"
        cached = tmp_path / "air-passengers.csv"
        cached.write_bytes(b"corrupted data")

        with pytest.raises(ValueError, match="[Hh]ash mismatch"):
            load_dataset("air-passengers", cache_dir=tmp_path)

    def test_valid_hash_returns_dataframe(self, tmp_path):
        csv_content = b"a,b\n1,2\n3,4\n"
        cached = tmp_path / "test.csv"
        cached.write_bytes(csv_content)

        real_hash = hashlib.sha256(csv_content).hexdigest()

        with patch.dict(
            "polars_ts.datasets.REGISTRY",
            {"test-csv": {"url": "https://example.com/test.csv", "sha256": real_hash}},
        ):
            result = load_dataset("test-csv", cache_dir=tmp_path)
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (2, 2)
