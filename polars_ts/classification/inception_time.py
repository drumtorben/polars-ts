"""InceptionTime classifier for time series.

Implements the InceptionTime architecture (Fawaz et al., 2020) —
an ensemble of Inception modules with residual connections for
time series classification.

References
----------
Fawaz et al. (2020). *InceptionTime: Finding AlexNet for Time
Series Classification.* Data Mining and Knowledge Discovery.

"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.classification._dl_utils import extract_classification_data


class _InceptionModule(nn.Module):
    """Single Inception module with multi-scale 1D convolutions."""

    def __init__(self, in_channels: int, n_filters: int) -> None:
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        # Multi-scale convolutions
        self.conv10 = nn.Conv1d(n_filters, n_filters, kernel_size=10, padding=5, bias=False)
        self.conv20 = nn.Conv1d(n_filters, n_filters, kernel_size=20, padding=10, bias=False)
        self.conv40 = nn.Conv1d(n_filters, n_filters, kernel_size=40, padding=20, bias=False)

        # Max pooling branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        c10 = self.conv10(bottleneck)
        c20 = self.conv20(bottleneck)
        c40 = self.conv40(bottleneck)
        pool = self.conv_pool(self.maxpool(x))

        # Trim to same length (padding may add 1 extra)
        min_len = min(c10.shape[2], c20.shape[2], c40.shape[2], pool.shape[2])
        out = torch.cat([c10[:, :, :min_len], c20[:, :, :min_len], c40[:, :, :min_len], pool[:, :, :min_len]], dim=1)
        return self.relu(self.bn(out))


class _InceptionBlock(nn.Module):
    """Inception block with residual connection."""

    def __init__(self, in_channels: int, n_filters: int) -> None:
        super().__init__()
        out_channels = n_filters * 4
        self.inception1 = _InceptionModule(in_channels, n_filters)
        self.inception2 = _InceptionModule(out_channels, n_filters)
        self.inception3 = _InceptionModule(out_channels, n_filters)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)
        # Trim residual to match
        res = self.residual(x)
        min_len = min(out.shape[2], res.shape[2])
        return self.relu(out[:, :, :min_len] + res[:, :, :min_len])


class _InceptionTimeNet(nn.Module):
    """Full InceptionTime network."""

    def __init__(self, n_classes: int, n_filters: int = 32) -> None:
        super().__init__()
        self.block = _InceptionBlock(1, n_filters)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        out = self.block(x)
        out = self.gap(out).squeeze(-1)
        return self.fc(out)


class InceptionTimeClassifier:
    """InceptionTime time series classifier.

    Parameters
    ----------
    n_filters
        Number of filters per Inception module branch.
    max_epochs
        Maximum training epochs.
    lr
        Learning rate.
    batch_size
        Training batch size.
    id_col, target_col, time_col
        Column names.

    """

    def __init__(
        self,
        n_filters: int = 32,
        max_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        id_col: str = "unique_id",
        target_col: str = "y",
        time_col: str = "ds",
    ) -> None:
        self.n_filters = n_filters
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self._model: _InceptionTimeNet | None = None
        self._label_encoder: dict[str, int] = {}
        self._label_decoder: dict[int, str] = {}
        self._mean: float = 0.0
        self._std: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame, *, label_col: str = "label") -> InceptionTimeClassifier:
        """Train on labeled time series data."""
        ids, X, y = extract_classification_data(
            df,
            self.id_col,
            self.target_col,
            self.time_col,
            label_col,
        )
        if y is None:
            raise ValueError(f"Label column {label_col!r} not found.")

        # Build label maps
        labels_df = df.group_by(self.id_col).agg(pl.col(label_col).first())
        unique_labels = sorted(labels_df[label_col].unique().to_list())
        self._label_encoder = {str(lbl): i for i, lbl in enumerate(unique_labels)}
        self._label_decoder = {i: str(lbl) for lbl, i in self._label_encoder.items()}
        n_classes = len(unique_labels)

        # Normalize
        self._mean = float(np.mean(X))
        self._std = float(np.std(X)) or 1.0
        X_norm = (X - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)  # (N, 1, T)
        y_t = torch.tensor(y, dtype=torch.long)

        model = _InceptionTimeNet(n_classes=n_classes, n_filters=self.n_filters)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict class labels for test series."""
        if not self.is_fitted_ or self._model is None:
            raise RuntimeError("Call fit() before predict().")

        ids, X, _ = extract_classification_data(
            df,
            self.id_col,
            self.target_col,
            self.time_col,
        )

        X_norm = (X - self._mean) / self._std
        X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            preds = logits.argmax(dim=1).numpy()

        labels = [self._label_decoder[int(p)] for p in preds]
        return pl.DataFrame({self.id_col: ids, "predicted_label": labels})
