"""N-BEATS: Neural Basis Expansion Analysis for Time Series.

Implements the N-BEATS architecture (Oreshkin et al., ICLR 2020) with
both generic and interpretable (trend + seasonality) stack types.

References
----------
Oreshkin et al. (2020). *N-BEATS: Neural basis expansion analysis
for interpretable time series forecasting.* ICLR.

"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.adapters.embeddings import _extract_series
from polars_ts.dl._training import build_forecast_df, build_windows


class _GenericBlock(nn.Module):
    """Generic N-BEATS block with learnable basis."""

    def __init__(self, input_size: int, h: int, hidden_size: int, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        self.fc = nn.Sequential(*layers)
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.backcast_fc(h), self.forecast_fc(h)


class _TrendBlock(nn.Module):
    """Interpretable trend block using polynomial basis."""

    def __init__(self, input_size: int, h: int, hidden_size: int, degree: int = 3) -> None:
        super().__init__()
        self.degree = degree
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(3):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, degree + 1)
        self.theta_f = nn.Linear(hidden_size, degree + 1)

        # Polynomial basis
        t_back = torch.linspace(0, 1, input_size).unsqueeze(0)
        t_fore = torch.linspace(0, 1, h).unsqueeze(0)
        powers = torch.arange(degree + 1, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("basis_back", t_back.pow(powers))  # (degree+1, input_size)
        self.register_buffer("basis_fore", t_fore.pow(powers))  # (degree+1, h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        tb = self.theta_b(h)  # (batch, degree+1)
        tf = self.theta_f(h)
        backcast = torch.einsum("bd,di->bi", tb, self.basis_back)
        forecast = torch.einsum("bd,di->bi", tf, self.basis_fore)
        return backcast, forecast


class _SeasonalityBlock(nn.Module):
    """Interpretable seasonality block using Fourier basis."""

    def __init__(self, input_size: int, h: int, hidden_size: int, n_harmonics: int = 5) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        n_coeffs = 2 * n_harmonics
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(3):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, n_coeffs)
        self.theta_f = nn.Linear(hidden_size, n_coeffs)

        # Fourier basis
        t_back = torch.linspace(0, 1, input_size).unsqueeze(0)
        t_fore = torch.linspace(0, 1, h).unsqueeze(0)
        freqs = torch.arange(1, n_harmonics + 1, dtype=torch.float32).unsqueeze(1) * 2 * torch.pi
        cos_back = torch.cos(freqs * t_back)
        sin_back = torch.sin(freqs * t_back)
        cos_fore = torch.cos(freqs * t_fore)
        sin_fore = torch.sin(freqs * t_fore)
        self.register_buffer("basis_back", torch.cat([cos_back, sin_back], dim=0))
        self.register_buffer("basis_fore", torch.cat([cos_fore, sin_fore], dim=0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        tb = self.theta_b(h)
        tf = self.theta_f(h)
        backcast = torch.einsum("bd,di->bi", tb, self.basis_back)
        forecast = torch.einsum("bd,di->bi", tf, self.basis_fore)
        return backcast, forecast


_BLOCK_TYPES = {
    "generic": _GenericBlock,
    "trend": _TrendBlock,
    "seasonality": _SeasonalityBlock,
}


class _NBEATSNet(nn.Module):
    """Full N-BEATS network."""

    def __init__(
        self,
        input_size: int,
        h: int,
        hidden_size: int,
        stack_types: list[str],
        n_blocks: int,
    ) -> None:
        super().__init__()
        self.h = h
        self.blocks = nn.ModuleList()
        for stype in stack_types:
            block_cls = _BLOCK_TYPES[stype]
            for _ in range(n_blocks):
                self.blocks.append(block_cls(input_size, h, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = torch.zeros(x.shape[0], self.h, device=x.device)
        residual = x
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast


class NBEATSForecaster:
    """N-BEATS time series forecaster.

    Parameters
    ----------
    h
        Forecast horizon.
    input_size
        Lookback window size.
    hidden_size
        Hidden layer size in each block.
    n_stacks
        Number of stacks (only used with default stack_types).
    n_blocks
        Number of blocks per stack.
    stack_types
        List of stack types: ``"generic"``, ``"trend"``, ``"seasonality"``.
        If ``None``, uses ``n_stacks`` generic stacks.
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
        input_size: int = 36,
        hidden_size: int = 128,
        n_stacks: int = 2,
        n_blocks: int = 3,
        stack_types: list[str] | None = None,
        max_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.stack_types = stack_types or ["generic"] * n_stacks
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._model: _NBEATSNet | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> NBEATSForecaster:
        """Train the N-BEATS model on historical data.

        Parameters
        ----------
        df
            Panel DataFrame with historical observations.

        Returns
        -------
        NBEATSForecaster
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

        model = _NBEATSNet(
            input_size=self.input_size,
            h=self.h,
            hidden_size=self.hidden_size,
            stack_types=self.stack_types,
            n_blocks=self.n_blocks,
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
                    # Pad short series with leading zeros
                    padded = np.zeros(self.input_size, dtype=np.float64)
                    padded[-len(arr) :] = arr.astype(np.float64)
                    context = padded
                else:
                    context = arr[-self.input_size :].astype(np.float64)
                x = torch.tensor((context - self._mean) / self._std, dtype=torch.float32).unsqueeze(0)
                pred = self._model(x).squeeze(0).numpy()
                all_forecasts[i] = pred * self._std + self._mean

        return build_forecast_df(ids, all_forecasts, df, self.h, self.id_col, self.time_col)
