"""Benchmarks for k-medoids PAM clustering."""

import polars as pl
import pytest

from polars_ts._distance_dispatch import compute_distances, pairwise_to_dict
from polars_ts.clustering.kmedoids import _build_dist_matrix, _kmedoids_python


def _make_cluster_data(n_series: int = 20, series_len: int = 50, seed: int = 42) -> pl.DataFrame:
    """Generate n_series time series with 2 natural clusters."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_series):
        sid = f"S{i:03d}"
        if i < n_series // 2:
            values = rng.normal(0, 1, series_len).cumsum().tolist()
        else:
            values = rng.normal(5, 1, series_len).cumsum().tolist()
        for v in values:
            rows.append({"unique_id": sid, "y": v})
    return pl.DataFrame(rows)


def _precompute_distances(df: pl.DataFrame) -> tuple[dict, list[str]]:
    """Precompute pairwise distances and return dict + sorted ids."""
    pairwise = compute_distances(df, df, method="dtw")
    dist_dict = pairwise_to_dict(pairwise)
    str_ids = [str(s) for s in df["unique_id"].unique().sort().to_list()]
    for sid in str_ids:
        dist_dict[(sid, sid)] = 0.0
    return dist_dict, str_ids


@pytest.fixture(scope="module")
def data_20():
    df = _make_cluster_data(n_series=20, series_len=30)
    dist_dict, str_ids = _precompute_distances(df)
    return dist_dict, str_ids


@pytest.fixture(scope="module")
def data_50():
    df = _make_cluster_data(n_series=50, series_len=30)
    dist_dict, str_ids = _precompute_distances(df)
    return dist_dict, str_ids


def test_kmedoids_rust_20(benchmark, data_20):
    from polars_ts_rs import kmedoids_pam

    dist_dict, str_ids = data_20
    flat = _build_dist_matrix(dist_dict, str_ids)
    n = len(str_ids)
    result = benchmark(kmedoids_pam, flat, n, 2, 100, 42)
    assert len(result[1]) == n


def test_kmedoids_python_20(benchmark, data_20):
    dist_dict, str_ids = data_20
    result = benchmark(_kmedoids_python, dict(dist_dict), str_ids, 2, 100, 42)
    assert len(result[1]) == len(str_ids)


def test_kmedoids_rust_50(benchmark, data_50):
    from polars_ts_rs import kmedoids_pam

    dist_dict, str_ids = data_50
    flat = _build_dist_matrix(dist_dict, str_ids)
    n = len(str_ids)
    result = benchmark(kmedoids_pam, flat, n, 3, 100, 42)
    assert len(result[1]) == n


def test_kmedoids_python_50(benchmark, data_50):
    dist_dict, str_ids = data_50
    result = benchmark(_kmedoids_python, dict(dist_dict), str_ids, 3, 100, 42)
    assert len(result[1]) == len(str_ids)
