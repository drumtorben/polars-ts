"""PatchTST: Patch-based Transformer for Time Series.

Implements a channel-independent PatchTST forecaster (Nie et al.,
ICLR 2023). The input is split into fixed-length patches, each
projected to a token embedding, then processed by a standard
transformer encoder. The final representation is linearly mapped
to the forecast horizon.

References
----------
Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers.* ICLR.

"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.adapters.embeddings import _extract_series
from polars_ts.dl._training import build_forecast_df, build_windows


class _PatchTSTNet(nn.Module):
    """PatchTST network."""

    def __init__(
        self,
        input_size: int,
        h: int,
        patch_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.h = h
        self.patch_len = patch_len
        self.d_model = d_model

        # Number of patches (with stride = patch_len)
        if input_size % patch_len != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by " f"patch_len ({patch_len}).")
        self.n_patches = input_size // patch_len

        # Patch embedding
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        d_ff = d_ff or d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten + linear head
        self.head = nn.Linear(self.n_patches * d_model, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, input_size)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(batch, h)``.

        """
        batch = x.shape[0]

        # Create patches: (batch, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)

        # Project patches to d_model
        tokens = self.patch_proj(patches)  # (batch, n_patches, d_model)
        tokens = tokens + self.pos_embed

        # Transformer encoder
        encoded = self.encoder(tokens)  # (batch, n_patches, d_model)

        # Flatten and project to forecast
        flat = encoded.reshape(batch, -1)
        return self.head(flat)


class PatchTSTForecaster:
    """PatchTST time series forecaster.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window size. Must be divisible by ``patch_len``.
    patch_len
        Length of each patch.
    d_model
        Transformer embedding dimension.
    n_heads
        Number of attention heads.
    n_layers
        Number of transformer encoder layers.
    d_ff
        Feedforward dimension. Defaults to ``4 * d_model``.
    dropout
        Dropout rate.
    max_epochs
        Maximum training epochs.
    lr
        Learning rate.
    batch_size
        Training batch size.
    id_col, time_col, target_col
        Column names.

    """

    def __init__(
        self,
        h: int = 12,
        input_size: int = 64,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._model: _PatchTSTNet | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> PatchTSTForecaster:
        """Train the PatchTST model on historical data.

        Parameters
        ----------
        df
            Panel DataFrame with historical observations.

        Returns
        -------
        PatchTSTForecaster
            Self, for chaining.

        """
        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)
        X, Y = build_windows(arrays, self.input_size, self.h)

        if len(X) == 0:
            raise ValueError(
                f"No training windows could be created. "
                f"Ensure series have at least {self.input_size + self.h} observations."
            )

        # Normalize
        self._mean = float(np.mean(X))
        self._std = float(np.std(X)) or 1.0
        X_norm = (X - self._mean) / self._std
        Y_norm = (Y - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32)

        model = _PatchTSTNet(
            input_size=self.input_size,
            h=self.h,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate forecasts for each series.

        Parameters
        ----------
        df
            Panel DataFrame (uses last ``input_size`` observations per series).

        Returns
        -------
        pl.DataFrame
            Columns: ``[id_col, time_col, y_hat]``.

        """
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict().")

        ids, arrays = _extract_series(df, self.target_col, self.id_col, self.time_col)

        self._model.eval()
        all_forecasts = np.zeros((len(ids), self.h))

        with torch.no_grad():
            for i, arr in enumerate(arrays):
                if len(arr) < self.input_size:
                    padded = np.zeros(self.input_size, dtype=np.float64)
                    padded[-len(arr) :] = arr.astype(np.float64)
                    context = padded
                else:
                    context = arr[-self.input_size :].astype(np.float64)
                x = torch.tensor((context - self._mean) / self._std, dtype=torch.float32).unsqueeze(0)
                pred = self._model(x).squeeze(0).numpy()
                all_forecasts[i] = pred * self._std + self._mean

        return build_forecast_df(ids, all_forecasts, df, self.h, self.id_col, self.time_col)
