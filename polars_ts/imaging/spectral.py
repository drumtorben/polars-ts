"""Spectrogram and wavelet scalogram imaging for time series.

Converts time series to time-frequency 2D representations via
Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_ts.imaging._utils import extract_series


def _spectrogram(
    x: np.ndarray,
    nperseg: int,
    noverlap: int | None,
    window: str,
    log_scale: bool,
) -> np.ndarray:
    """Compute STFT spectrogram for a single 1D series."""
    from scipy.signal import stft

    if noverlap is None:
        noverlap = nperseg // 2

    _, _, Zxx = stft(x, nperseg=nperseg, noverlap=noverlap, window=window)
    mag = np.abs(Zxx)

    if log_scale:
        mag = np.log1p(mag)

    return mag


def _morlet(M: int, s: float = 1.0, w: float = 5.0) -> np.ndarray:
    """Complex Morlet wavelet."""
    t = np.arange(-M // 2, M // 2 + 1, dtype=np.float64)
    output = np.exp(1j * w * t / s) * np.exp(-0.5 * (t / s) ** 2) * np.pi ** (-0.25)
    return output


def _ricker(M: int, a: float = 1.0) -> np.ndarray:
    """Mexican hat (Ricker) wavelet."""
    t = np.arange(-M // 2, M // 2 + 1, dtype=np.float64) / a
    return (2.0 / (np.sqrt(3 * a) * np.pi**0.25)) * (1 - t**2) * np.exp(-0.5 * t**2)


def _cwt(x: np.ndarray, wavelet_func: str, scales: np.ndarray) -> np.ndarray:
    """Continuous Wavelet Transform using convolution."""
    from scipy.signal import fftconvolve

    n = len(x)
    coeffs = np.zeros((len(scales), n), dtype=np.complex128)

    for i, scale in enumerate(scales):
        M = min(10 * int(np.ceil(scale)), n)
        if M < 1:
            M = 1
        if wavelet_func in ("morlet", "morl"):
            wavelet = _morlet(M, s=scale)
        else:
            wavelet = _ricker(M, a=scale)
        # Convolve and trim to original length
        conv = fftconvolve(x, wavelet[::-1].conj(), mode="same")
        coeffs[i] = conv

    return coeffs


def _scalogram(
    x: np.ndarray,
    wavelet: str,
    scales: np.ndarray,
) -> np.ndarray:
    """Compute CWT scalogram for a single 1D series."""
    supported = ("morlet", "morl", "ricker", "mexh")
    # Normalise aliases
    wname = wavelet
    if wname == "mexh":
        wname = "ricker"
    if wname not in supported:
        raise ValueError(f"Unknown wavelet {wavelet!r}. Supported: {list(supported)}")

    coeffs = _cwt(x, wname, scales)
    return np.abs(coeffs)


def to_spectrogram(
    df: pl.DataFrame,
    nperseg: int = 64,
    noverlap: int | None = None,
    window: str = "hann",
    log_scale: bool = True,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to STFT spectrogram images.

    Computes the Short-Time Fourier Transform magnitude for each series.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    nperseg
        Length of each STFT segment.
    noverlap
        Number of overlapping points between segments.
        Defaults to ``nperseg // 2``.
    window
        Window function name (e.g. ``"hann"``, ``"hamming"``).
    log_scale
        Apply ``log1p`` to the magnitude for better dynamic range.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a 2D array (frequency x time bins).

    """
    series = extract_series(df, id_col, target_col)
    return {sid: _spectrogram(vals, nperseg, noverlap, window, log_scale) for sid, vals in series.items()}


def to_scalogram(
    df: pl.DataFrame,
    wavelet: str = "morlet",
    scales: np.ndarray | None = None,
    n_scales: int = 32,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> dict[str, np.ndarray]:
    """Convert time series to CWT scalogram images.

    Computes the Continuous Wavelet Transform magnitude for each series.

    Parameters
    ----------
    df
        DataFrame with columns ``id_col`` and ``target_col``.
    wavelet
        Wavelet name: ``"morlet"`` / ``"morl"`` or ``"ricker"`` / ``"mexh"``.
    scales
        Array of scales to use. If ``None``, generates ``n_scales``
        logarithmically spaced scales from 1 to series_length / 4.
    n_scales
        Number of scales when ``scales`` is ``None``.
    id_col
        Column identifying each time series.
    target_col
        Column with the time series values.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from series ID to a 2D array (scale x time).

    """
    series = extract_series(df, id_col, target_col)
    result: dict[str, np.ndarray] = {}
    for sid, vals in series.items():
        if scales is None:
            max_scale = max(len(vals) // 4, 2)
            auto_scales = np.geomspace(1, max_scale, num=n_scales)
        else:
            auto_scales = scales
        result[sid] = _scalogram(vals, wavelet, auto_scales)
    return result
