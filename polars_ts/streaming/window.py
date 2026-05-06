"""Memory-efficient sliding window state manager."""

from __future__ import annotations

from collections import deque

import numpy as np


class SlidingWindowManager:
    """Maintains a fixed-size sliding window of observations per series.

    Parameters
    ----------
    window_size
        Maximum number of observations to retain per series.

    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self._buffers: dict[str, deque[float]] = {}

    def append(self, series_id: str, values: np.ndarray) -> None:
        """Append new observations for a series, evicting old ones."""
        if series_id not in self._buffers:
            self._buffers[series_id] = deque(maxlen=self.window_size)
        buf = self._buffers[series_id]
        arr = np.asarray(values, dtype=np.float64).ravel()
        # Optimize: if incoming data exceeds window, only keep the tail
        if len(arr) >= self.window_size:
            buf.clear()
            arr = arr[-self.window_size :]
        for v in arr:
            buf.append(v)

    def get(self, series_id: str) -> np.ndarray:
        """Return current window contents for a series."""
        if series_id not in self._buffers:
            return np.array([], dtype=np.float64)
        return np.array(self._buffers[series_id], dtype=np.float64)

    def clear(self, series_id: str) -> None:
        """Clear buffer for a specific series."""
        if series_id in self._buffers:
            self._buffers[series_id].clear()

    def clear_all(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()

    @property
    def series_ids(self) -> list[str]:
        """Return list of tracked series IDs."""
        return list(self._buffers.keys())
