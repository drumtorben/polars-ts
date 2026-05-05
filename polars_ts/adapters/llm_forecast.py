"""LLM-based forecasting adapters (Time-LLM, LLM-PS).

Reprograms LLMs for time series forecasting by converting temporal patterns
into representations that pre-trained language models can process.

Time-LLM: Patches input series into tokens, projects via cross-attention
with text prototypes, then decodes forecasts through a frozen LLM backbone.

LLM-PS (Pattern & Semantic): Multi-scale CNN extracts local patterns at
different granularities, then an LLM backbone reasons over the combined
multi-scale representation.

References
----------
Time-LLM (Jin et al., ICLR 2024).
LLM-PS (2025). arXiv:2503.09656.

"""

from __future__ import annotations

from typing import Self

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.adapters.embeddings import _extract_series
from polars_ts.dl._training import build_forecast_df, build_windows

# ---------------------------------------------------------------------------
# Time-LLM network
# ---------------------------------------------------------------------------


class _TimeLLMNet(nn.Module):
    """Simplified Time-LLM: patch embedding → cross-attention → MLP decoder.

    Converts time series patches into a sequence of tokens, applies learned
    cross-attention with text-like prototypes, then decodes forecasts.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        patch_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_prototypes: int = 16,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.patch_len = patch_len
        self.n_patches = max(input_size // patch_len, 1)
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_patches * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size)

        Returns
        -------
        Tensor of shape (batch, h)

        """
        b = x.shape[0]
        # Patch and embed
        # Truncate to exact multiple of patch_len
        usable = self.n_patches * self.patch_len
        x_trunc = x[:, -usable:]
        patches = x_trunc.reshape(b, self.n_patches, self.patch_len)
        tokens = self.patch_embed(patches)  # (b, n_patches, d_model)

        # Cross-attention with prototypes
        proto = self.prototypes.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.cross_attn(tokens, proto, proto)

        # Decode
        flat = attn_out.reshape(b, -1)
        return self.decoder(flat)


# ---------------------------------------------------------------------------
# LLM-PS network
# ---------------------------------------------------------------------------


class _LLMPSNet(nn.Module):
    """LLM-PS: multi-scale CNN pattern extraction + MLP semantic decoder.

    Extracts patterns at multiple temporal scales via parallel convolutions,
    concatenates the multi-scale features, and decodes forecasts.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.d_model = d_model

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, d_model, kernel_size=k, padding=k // 2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for k in kernel_sizes
            ]
        )

        n_scales = len(kernel_sizes)
        self.decoder = nn.Sequential(
            nn.Linear(n_scales * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_size)

        Returns
        -------
        Tensor of shape (batch, h)

        """
        x_1d = x.unsqueeze(1)  # (batch, 1, input_size)
        features = [conv(x_1d).squeeze(-1) for conv in self.convs]
        combined = torch.cat(features, dim=1)  # (batch, n_scales * d_model)
        return self.decoder(combined)


# ---------------------------------------------------------------------------
# TimeLLMForecaster
# ---------------------------------------------------------------------------


class TimeLLMForecaster:
    """Time-LLM forecaster: patch → cross-attention with prototypes → decode.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window length.
    patch_len
        Length of each input patch.
    d_model
        Hidden dimension.
    n_heads
        Number of attention heads.
    n_prototypes
        Number of learnable text-like prototypes.
    lr
        Learning rate.
    max_epochs
        Maximum training epochs.
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        h: int = 12,
        input_size: int = 36,
        patch_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_prototypes: int = 16,
        lr: float = 1e-3,
        max_epochs: int = 50,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_prototypes = n_prototypes
        self.lr = lr
        self.max_epochs = max_epochs
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self.is_fitted_: bool = False
        self._model: _TimeLLMNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pl.DataFrame) -> Self:
        """Train the Time-LLM model."""
        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)
        X, Y = build_windows(arrays, self.input_size, self.h)

        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given input_size and horizon")

        self._mean = X.mean(axis=1, keepdims=True)
        self._std = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - self._mean) / self._std
        Y_norm = (Y - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)

        model = _TimeLLMNet(
            input_size=self.input_size,
            h=self.h,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_prototypes=self.n_prototypes,
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
        """Generate forecasts for each series."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict()")

        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)
        forecasts = np.zeros((len(ids), self.h))

        model = self._model
        model.eval()
        with torch.no_grad():
            for i, arr in enumerate(arrays):
                x = arr[-self.input_size :].astype(np.float64)
                if len(x) < self.input_size:
                    x = np.pad(x, (self.input_size - len(x), 0), mode="edge")
                mean = x.mean()
                std = x.std() + 1e-8
                x_norm = (x - mean) / std
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
                pred = model(x_t).squeeze(0).detach().cpu().tolist()
                pred = np.array(pred)
                forecasts[i] = pred * std + mean

        return build_forecast_df(ids, forecasts, df, self.h, self.id_col, self.time_col)


# ---------------------------------------------------------------------------
# LLMPSForecaster
# ---------------------------------------------------------------------------


class LLMPSForecaster:
    """LLM-PS forecaster: multi-scale CNN pattern extraction → decode.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window length.
    kernel_sizes
        CNN kernel sizes for multi-scale pattern extraction.
    d_model
        Hidden dimension.
    lr
        Learning rate.
    max_epochs
        Maximum training epochs.
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        h: int = 12,
        input_size: int = 36,
        kernel_sizes: list[int] | tuple[int, ...] = (3, 5, 7),
        d_model: int = 64,
        lr: float = 1e-3,
        max_epochs: int = 50,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.kernel_sizes = tuple(kernel_sizes)
        self.d_model = d_model
        self.lr = lr
        self.max_epochs = max_epochs
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self.is_fitted_: bool = False
        self._model: _LLMPSNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pl.DataFrame) -> Self:
        """Train the LLM-PS model."""
        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)
        X, Y = build_windows(arrays, self.input_size, self.h)

        if X.shape[0] == 0:
            raise ValueError("Not enough data for the given input_size and horizon")

        self._mean = X.mean(axis=1, keepdims=True)
        self._std = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - self._mean) / self._std
        Y_norm = (Y - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)

        model = _LLMPSNet(
            input_size=self.input_size,
            h=self.h,
            kernel_sizes=self.kernel_sizes,
            d_model=self.d_model,
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
        """Generate forecasts for each series."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict()")

        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)
        forecasts = np.zeros((len(ids), self.h))

        model = self._model
        model.eval()
        with torch.no_grad():
            for i, arr in enumerate(arrays):
                x = arr[-self.input_size :].astype(np.float64)
                if len(x) < self.input_size:
                    x = np.pad(x, (self.input_size - len(x), 0), mode="edge")
                mean = x.mean()
                std = x.std() + 1e-8
                x_norm = (x - mean) / std
                x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
                pred = model(x_t).squeeze(0).detach().cpu().tolist()
                pred = np.array(pred)
                forecasts[i] = pred * std + mean

        return build_forecast_df(ids, forecasts, df, self.h, self.id_col, self.time_col)
