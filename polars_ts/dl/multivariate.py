"""Native multivariate deep forecasting models.

MultivariatePatchTST: Channel-mixing PatchTST that processes all variates
jointly through a shared transformer encoder.

iTransformerForecaster: Inverted transformer that treats each variate as
a token (Liu et al., 2024), enabling cross-variate attention.

References
----------
Nie et al. (2023). PatchTST. ICLR.
Liu et al. (2024). iTransformer. ICLR.

"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.models.baselines import _infer_freq, _make_future_dates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_multivariate(
    df: pl.DataFrame,
    target_cols: list[str],
    id_col: str,
    time_col: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Extract per-series multivariate arrays.

    Returns
    -------
    ids
        List of series identifiers.
    arrays
        List of arrays, each of shape ``(seq_len, n_vars)``.

    """
    sorted_df = df.sort(id_col, time_col)
    ids: list[str] = []
    arrays: list[np.ndarray] = []
    for key, group in sorted_df.group_by(id_col, maintain_order=True):
        ids.append(str(key[0] if isinstance(key, tuple) else key))
        cols = [group[c].to_numpy().astype(np.float64) for c in target_cols]
        arrays.append(np.column_stack(cols))  # (seq_len, n_vars)
    return ids, arrays


def _build_mv_windows(
    arrays: list[np.ndarray],
    input_size: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build multivariate sliding windows.

    Returns
    -------
    X
        Shape ``(n_windows, input_size, n_vars)``.
    Y
        Shape ``(n_windows, h, n_vars)``.

    """
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for arr in arrays:
        n = len(arr)
        for t in range(input_size, n - h + 1):
            all_x.append(arr[t - input_size : t])
            all_y.append(arr[t : t + h])
    if not all_x:
        n_vars = arrays[0].shape[1] if arrays else 1
        return np.empty((0, input_size, n_vars)), np.empty((0, h, n_vars))
    return np.array(all_x), np.array(all_y)


def _build_mv_forecast_df(
    ids: list[str],
    forecasts: np.ndarray,
    target_cols: list[str],
    df: pl.DataFrame,
    h: int,
    id_col: str,
    time_col: str,
) -> pl.DataFrame:
    """Build output DataFrame with future dates and multivariate forecasts.

    Parameters
    ----------
    forecasts
        Shape ``(n_series, h, n_vars)``.

    """
    sorted_df = df.sort(id_col, time_col)
    rows: list[dict[str, Any]] = []

    for i, sid in enumerate(ids):
        series_df = sorted_df.filter(pl.col(id_col) == sid)
        times = series_df[time_col]
        freq = _infer_freq(times)
        last_time = times[-1]
        future_dates = _make_future_dates(last_time, freq, h)

        for t in range(h):
            row: dict[str, Any] = {id_col: sid, time_col: future_dates[t]}
            for v, col in enumerate(target_cols):
                row[f"{col}_hat"] = float(forecasts[i, t, v])
            rows.append(row)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# MultivariatePatchTST network
# ---------------------------------------------------------------------------


class _MVPatchTSTNet(nn.Module):
    """Channel-mixing PatchTST: all variates processed jointly."""

    def __init__(
        self,
        input_size: int,
        h: int,
        n_vars: int,
        patch_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.n_patches = max(input_size // patch_len, 1)

        # Project each patch (patch_len * n_vars) → d_model
        self.patch_proj = nn.Linear(patch_len * n_vars, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, h * n_vars)
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size, n_vars)

        Returns
        -------
        Tensor of shape (batch, h, n_vars)

        """
        b = x.shape[0]
        usable = self.n_patches * self.patch_len
        x_trunc = x[:, -usable:, :]  # (b, usable, n_vars)

        # Reshape to patches: (b, n_patches, patch_len, n_vars)
        patches = x_trunc.reshape(b, self.n_patches, self.patch_len, self.n_vars)
        # Flatten per patch: (b, n_patches, patch_len * n_vars)
        patches_flat = patches.reshape(b, self.n_patches, -1)

        tokens = self.patch_proj(patches_flat) + self.pos_embed
        encoded = self.encoder(tokens)
        flat = encoded.reshape(b, -1)
        out = self.head(flat)
        return out.reshape(b, self.h, self.n_vars)


# ---------------------------------------------------------------------------
# iTransformer network
# ---------------------------------------------------------------------------


class _iTransformerNet(nn.Module):
    """Inverted transformer: each variate is a token."""

    def __init__(
        self,
        input_size: int,
        h: int,
        n_vars: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.h = h

        # Project each variate's full history → d_model
        self.variate_proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size, n_vars)

        Returns
        -------
        Tensor of shape (batch, h, n_vars)

        """
        # Transpose to (batch, n_vars, input_size) — each variate is a token
        x_t = x.transpose(1, 2)
        tokens = self.variate_proj(x_t) + self.pos_embed  # (b, n_vars, d_model)
        encoded = self.encoder(tokens)  # (b, n_vars, d_model)
        out = self.head(encoded)  # (b, n_vars, h)
        return out.transpose(1, 2)  # (b, h, n_vars)


# ---------------------------------------------------------------------------
# MultivariatePatchTST Forecaster
# ---------------------------------------------------------------------------


class MultivariatePatchTST:
    """Channel-mixing PatchTST for multivariate forecasting.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window length.
    patch_len
        Length of each input patch.
    target_cols
        List of target column names to forecast jointly.
    d_model
        Transformer hidden dimension.
    n_heads
        Number of attention heads.
    n_layers
        Number of transformer encoder layers.
    dropout
        Dropout rate.
    lr
        Learning rate.
    max_epochs
        Maximum training epochs.
    id_col, time_col
        Column names.

    """

    def __init__(
        self,
        h: int = 12,
        input_size: int = 32,
        patch_len: int = 8,
        target_cols: list[str] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 50,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.target_cols = target_cols or ["y"]
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.id_col = id_col
        self.time_col = time_col

        self.is_fitted_: bool = False
        self._model: _MVPatchTSTNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pl.DataFrame) -> Self:
        """Train on multivariate panel data."""
        ids, arrays = _extract_multivariate(df, self.target_cols, self.id_col, self.time_col)
        X, Y = _build_mv_windows(arrays, self.input_size, self.h)

        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given input_size and horizon")

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm = (X - self._mean) / self._std
        Y_norm = (Y - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)

        model = _MVPatchTSTNet(
            input_size=self.input_size,
            h=self.h,
            n_vars=len(self.target_cols),
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate multivariate forecasts."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict()")

        ids, arrays = _extract_multivariate(df, self.target_cols, self.id_col, self.time_col)
        n_vars = len(self.target_cols)
        forecasts = np.zeros((len(ids), self.h, n_vars))

        model = self._model
        model.eval()
        with torch.no_grad():
            for i, arr in enumerate(arrays):
                x = arr[-self.input_size :].astype(np.float64)
                if len(x) < self.input_size:
                    padded = np.zeros((self.input_size, n_vars), dtype=np.float64)
                    padded[-len(x) :] = x
                    x = padded
                x_norm = (x - self._mean.squeeze(0)) / self._std.squeeze(0)
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
                pred = model(x_t).squeeze(0).detach().cpu().tolist()
                pred = np.array(pred)
                forecasts[i] = pred * self._std.squeeze(0) + self._mean.squeeze(0)

        return _build_mv_forecast_df(ids, forecasts, self.target_cols, df, self.h, self.id_col, self.time_col)


# ---------------------------------------------------------------------------
# iTransformerForecaster
# ---------------------------------------------------------------------------


class iTransformerForecaster:
    """Inverted transformer forecaster for multivariate time series.

    Treats each variate as a token, enabling cross-variate attention.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window length.
    target_cols
        List of target column names to forecast jointly.
    d_model
        Transformer hidden dimension.
    n_heads
        Number of attention heads.
    n_layers
        Number of transformer encoder layers.
    dropout
        Dropout rate.
    lr
        Learning rate.
    max_epochs
        Maximum training epochs.
    id_col, time_col
        Column names.

    """

    def __init__(
        self,
        h: int = 12,
        input_size: int = 30,
        target_cols: list[str] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 50,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.target_cols = target_cols or ["y"]
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.id_col = id_col
        self.time_col = time_col

        self.is_fitted_: bool = False
        self._model: _iTransformerNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pl.DataFrame) -> Self:
        """Train on multivariate panel data."""
        ids, arrays = _extract_multivariate(df, self.target_cols, self.id_col, self.time_col)
        X, Y = _build_mv_windows(arrays, self.input_size, self.h)

        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given input_size and horizon")

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X_norm = (X - self._mean) / self._std
        Y_norm = (Y - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)

        model = _iTransformerNet(
            input_size=self.input_size,
            h=self.h,
            n_vars=len(self.target_cols),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate multivariate forecasts."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict()")

        ids, arrays = _extract_multivariate(df, self.target_cols, self.id_col, self.time_col)
        n_vars = len(self.target_cols)
        forecasts = np.zeros((len(ids), self.h, n_vars))

        model = self._model
        model.eval()
        with torch.no_grad():
            for i, arr in enumerate(arrays):
                x = arr[-self.input_size :].astype(np.float64)
                if len(x) < self.input_size:
                    padded = np.zeros((self.input_size, n_vars), dtype=np.float64)
                    padded[-len(x) :] = x
                    x = padded
                x_norm = (x - self._mean.squeeze(0)) / self._std.squeeze(0)
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
                pred = model(x_t).squeeze(0).detach().cpu().tolist()
                pred = np.array(pred)
                forecasts[i] = pred * self._std.squeeze(0) + self._mean.squeeze(0)

        return _build_mv_forecast_df(ids, forecasts, self.target_cols, df, self.h, self.id_col, self.time_col)
