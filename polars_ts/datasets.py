"""Dataset loading with SHA-256 integrity verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

REGISTRY: dict[str, dict[str, str]] = {
    "m4-hourly": {
        "url": "https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet",
        "sha256": "6e80d9435cbab4e8cf0ca42dcb45489350755e36471dc9c97083eb6511f47af8",
    },
    "air-passengers": {
        "url": "https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv",
        "sha256": "fe11b5863af07d7dbc81f7b3a661f43bc055ce79a914073b9f4e7a648db097f8",
    },
    "m5_y": {
        "url": "https://datasets-nixtla.s3.amazonaws.com/m5_y.parquet",
        "sha256": "c6607c61e406bc1212756df2e9405ca436d1ca1a482c4c57f79d3a6eb73e0a16",
    },
}


def _download(url: str, dest: Path) -> Path:
    """Download a file from *url* to *dest*."""
    import urllib.request

    urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


def load_dataset(
    name: str,
    *,
    cache_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Load a known dataset with SHA-256 integrity verification.

    Parameters
    ----------
    name
        Dataset key (e.g. ``"m4-hourly"``). See ``REGISTRY`` for available names.
    cache_dir
        Directory for caching downloads. Defaults to ``~/.cache/polars_ts``.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    ValueError
        If the downloaded file's hash does not match the expected hash.

    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset: {name!r}. Available: {sorted(REGISTRY)}")

    entry = REGISTRY[name]
    url = entry["url"]
    expected_hash = entry["sha256"]

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "polars_ts"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = url.rsplit("/", 1)[-1]
    local_path = cache_dir / filename

    if not local_path.exists():
        _download(url, local_path)

    actual_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        local_path.unlink(missing_ok=True)
        raise ValueError(f"Hash mismatch for {name!r}: expected {expected_hash[:16]}..., " f"got {actual_hash[:16]}...")

    if filename.endswith(".csv"):
        return pl.read_csv(local_path, try_parse_dates=True)
    return pl.read_parquet(local_path)
