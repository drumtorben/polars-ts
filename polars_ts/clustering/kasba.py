"""KASBA: K-means with Accelerated Stochastic Barycenter Averaging.

Uses MSM (Move-Split-Merge) elastic distance — a proper metric that
enables triangle inequality pruning for 10-30x speedup over aeon.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from polars_ts_rs.polars_ts_rs import kasba_fit, kasba_predict


class KASBAClusterer:
    """KASBA clustering for univariate and multivariate time series.

    Parameters
    ----------
    n_clusters
        Number of clusters. Default 8.
    c
        MSM cost coefficient. Default 1.0.
    independent
        If True, compute MSM per-channel independently (sum).
        If False, use dependent cross-channel MSM. Default True.
    ba_subset_size
        Proportion of cluster members used in barycenter SGD. Default 0.5.
    initial_step_size
        SGD initial learning rate. Default 0.05.
    decay_rate
        Exponential decay rate for step size. Default 0.1.
    max_iter
        Maximum k-means iterations. Default 10.
    seed
        Random seed for reproducibility. Default 42.

    """

    def __init__(
        self,
        n_clusters: int = 8,
        c: float = 1.0,
        independent: bool = True,
        ba_subset_size: float = 0.5,
        initial_step_size: float = 0.05,
        decay_rate: float = 0.1,
        max_iter: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.c = c
        self.independent = independent
        self.ba_subset_size = ba_subset_size
        self.initial_step_size = initial_step_size
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.seed = seed

        self.labels_: pl.DataFrame | None = None
        self.centroids_: np.ndarray | None = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self._is_fitted = False
        self._id_col: str = "unique_id"
        self._id_dtype: Any = None
        self._n_channels: int = 1
        self._ts_len: int = 0

    def fit(
        self,
        df: pl.DataFrame,
        id_col: str = "unique_id",
        target_col: str = "y",
        channel_col: str | None = None,
    ) -> KASBAClusterer:
        """Fit KASBA clustering.

        Parameters
        ----------
        df
            DataFrame with time series data.
        id_col
            Column identifying each time series.
        target_col
            Column with the time series values.
        channel_col
            Column identifying channels for multivariate series.
            If None, assumes univariate data.

        """
        self._id_col = id_col
        self._id_dtype = df[id_col].dtype

        data_3d, ids = self._to_3d_array(df, id_col, target_col, channel_col)

        labels_arr, centers, inertia, n_iter = kasba_fit(
            data_3d,
            n_clusters=self.n_clusters,
            c=self.c,
            independent=self.independent,
            ba_subset_size=self.ba_subset_size,
            initial_step_size=self.initial_step_size,
            decay_rate=self.decay_rate,
            n_iters=self.max_iter,
            random_seed=self.seed,
        )

        self.centroids_ = centers
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.labels_ = pl.DataFrame(
            {id_col: ids, "cluster": labels_arr.tolist()},
            schema={id_col: pl.String, "cluster": pl.Int64},
        ).with_columns(pl.col(id_col).cast(self._id_dtype))
        self._is_fitted = True
        return self

    def predict(
        self,
        df: pl.DataFrame,
        id_col: str = "unique_id",
        target_col: str = "y",
        channel_col: str | None = None,
    ) -> pl.DataFrame:
        """Predict cluster assignments for new data.

        Parameters
        ----------
        df
            DataFrame with time series data.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``[id_col, "cluster"]``.

        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        data_3d, ids = self._to_3d_array(df, id_col, target_col, channel_col)

        pred_labels = kasba_predict(
            self.centroids_,
            data_3d,
            c=self.c,
            independent=self.independent,
        )

        return pl.DataFrame(
            {id_col: ids, "cluster": pred_labels.tolist()},
            schema={id_col: pl.String, "cluster": pl.Int64},
        ).with_columns(pl.col(id_col).cast(self._id_dtype))

    def _to_3d_array(
        self,
        df: pl.DataFrame,
        id_col: str,
        target_col: str,
        channel_col: str | None,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert DataFrame to 3D numpy array (n_cases, n_channels, n_timepoints)."""
        ids = sorted(df[id_col].unique().cast(pl.String).to_list())
        n_cases = len(ids)

        if channel_col is None:
            # Univariate: (n_cases, 1, n_timepoints)
            series_list = []
            for uid in ids:
                vals = df.filter(pl.col(id_col).cast(pl.String) == uid)[target_col].to_numpy()
                series_list.append(vals.astype(np.float64))

            # Pad to max length
            max_len = max(len(s) for s in series_list)
            self._ts_len = max_len
            self._n_channels = 1

            data = np.zeros((n_cases, 1, max_len), dtype=np.float64)
            for i, s in enumerate(series_list):
                data[i, 0, : len(s)] = s
        else:
            # Multivariate: (n_cases, n_channels, n_timepoints)
            channels = sorted(df[channel_col].unique().cast(pl.String).to_list())
            n_channels = len(channels)
            self._n_channels = n_channels

            # Find max timepoints per (id, channel) combination
            max_len = 0
            for uid in ids:
                for ch in channels:
                    n = df.filter(
                        (pl.col(id_col).cast(pl.String) == uid) & (pl.col(channel_col).cast(pl.String) == ch)
                    ).height
                    max_len = max(max_len, n)
            self._ts_len = max_len

            data = np.zeros((n_cases, n_channels, max_len), dtype=np.float64)
            for i, uid in enumerate(ids):
                for j, ch in enumerate(channels):
                    vals = df.filter(
                        (pl.col(id_col).cast(pl.String) == uid) & (pl.col(channel_col).cast(pl.String) == ch)
                    )[target_col].to_numpy()
                    data[i, j, : len(vals)] = vals.astype(np.float64)

        return data, ids


def kasba(
    df: pl.DataFrame,
    k: int,
    *,
    c: float = 1.0,
    independent: bool = True,
    max_iter: int = 10,
    seed: int = 42,
    id_col: str = "unique_id",
    target_col: str = "y",
    channel_col: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """KASBA clustering convenience function.

    Parameters
    ----------
    df
        DataFrame with time series data.
    k
        Number of clusters.
    c
        MSM cost coefficient. Default 1.0.
    independent
        MSM mode. Default True.
    max_iter
        Maximum iterations. Default 10.
    seed
        Random seed. Default 42.
    id_col
        Series identifier column. Default ``"unique_id"``.
    target_col
        Values column. Default ``"y"``.
    channel_col
        Channel column for multivariate data. Default None.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``[id_col, "cluster"]``.

    """
    clf = KASBAClusterer(
        n_clusters=k,
        c=c,
        independent=independent,
        max_iter=max_iter,
        seed=seed,
        **kwargs,
    )
    clf.fit(df, id_col=id_col, target_col=target_col, channel_col=channel_col)
    assert clf.labels_ is not None
    return clf.labels_
