"""ResNet classifier for time series.

Implements a 1D ResNet adapted for time series classification
with residual blocks and global average pooling.

References
----------
Wang et al. (2017). *Time Series Classification from Scratch
with Deep Neural Networks: A Strong Baseline.* IJCNN.

"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from polars_ts.classification._dl_utils import extract_classification_data


class _ResBlock(nn.Module):
    """1D residual block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Trim to match
        res = self.shortcut(x)
        min_len = min(out.shape[2], res.shape[2])
        return self.relu(out[:, :, :min_len] + res[:, :, :min_len])


class _ResNet1D(nn.Module):
    """1D ResNet for time series classification."""

    def __init__(self, n_classes: int, n_filters: int = 64) -> None:
        super().__init__()
        self.block1 = _ResBlock(1, n_filters)
        self.block2 = _ResBlock(n_filters, n_filters * 2)
        self.block3 = _ResBlock(n_filters * 2, n_filters * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.gap(out).squeeze(-1)
        return self.fc(out)


class ResNetClassifier:
    """1D ResNet time series classifier.

    Parameters
    ----------
    n_filters
        Base number of convolutional filters.
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
        n_filters: int = 64,
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
        self._model: _ResNet1D | None = None
        self._label_encoder: dict[str, int] = {}
        self._label_decoder: dict[int, str] = {}
        self._mean: float = 0.0
        self._std: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, df: pl.DataFrame, *, label_col: str = "label") -> ResNetClassifier:
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

        labels_df = df.group_by(self.id_col).agg(pl.col(label_col).first())
        unique_labels = sorted(labels_df[label_col].unique().to_list())
        self._label_encoder = {str(lbl): i for i, lbl in enumerate(unique_labels)}
        self._label_decoder = {i: str(lbl) for lbl, i in self._label_encoder.items()}
        n_classes = len(unique_labels)

        self._mean = float(np.mean(X))
        self._std = float(np.std(X)) or 1.0
        X_norm = (X - self._mean) / self._std

        X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)

        model = _ResNet1D(n_classes=n_classes, n_filters=self.n_filters)
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
