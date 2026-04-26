# Time Series Imaging Guide

polars-ts can convert time series into 2D image representations, enabling the use of computer vision techniques for clustering, classification, and anomaly detection.

## Overview

All imaging functions follow the same API:

- **Input**: `pl.DataFrame` with `[unique_id, y]` columns
- **Output**: `dict[str, np.ndarray]` mapping series ID → 2D image array

| Function | Module | Output |
|---|---|---|
| `to_recurrence_plot` | `polars_ts.imaging.recurrence` | Binary/grayscale n×n distance matrix |
| `rqa_features` | `polars_ts.imaging.recurrence` | Dict of 6 scalar RQA features |
| `to_gasf` | `polars_ts.imaging.angular` | Gramian Angular Summation Field |
| `to_gadf` | `polars_ts.imaging.angular` | Gramian Angular Difference Field |
| `to_mtf` | `polars_ts.imaging.transition` | Markov Transition Field |
| `to_spectrogram` | `polars_ts.imaging.spectral` | STFT magnitude (freq × time) |
| `to_scalogram` | `polars_ts.imaging.spectral` | CWT magnitude (scale × time) |
| `extract_vision_embeddings` | `polars_ts.imaging.embeddings` | DataFrame of dense feature vectors |

## Recurrence plots

A recurrence plot is a matrix where `R[i,j] = 1` if observations `x(i)` and `x(j)` are within a threshold distance. Periodic signals produce diagonal lines; chaotic signals produce complex textures.

```python
from polars_ts import to_recurrence_plot, rqa_features

images = to_recurrence_plot(df, threshold=0.1, metric="euclidean", normalize=True)

# Extract quantitative features
for sid, rp in images.items():
    features = rqa_features(rp)
    print(f"{sid}: RR={features['recurrence_rate']:.3f}, DET={features['determinism']:.3f}")
```

**RQA features**: recurrence rate, determinism, laminarity, mean diagonal line length, mean vertical line length, entropy.

## Gramian Angular Fields

GAF encodes temporal correlations as angular differences on the unit circle.

```python
from polars_ts import to_gasf, to_gadf

gasf_images = to_gasf(df)                     # cos(phi_i + phi_j)
gadf_images = to_gadf(df)                     # sin(phi_i - phi_j)
gasf_small = to_gasf(df, image_size=64)       # PAA downsampled to 64×64
```

- **GASF** values lie in [-1, 1], symmetric matrix
- **GADF** values lie in [-1, 1], antisymmetric matrix (diagonal = 0)

## Markov Transition Fields

MTF discretises values into quantile bins and maps transition probabilities onto the time axis.

```python
from polars_ts import to_mtf

mtf_images = to_mtf(df, n_bins=8)             # 8 quantile bins
mtf_small = to_mtf(df, n_bins=8, image_size=64)
```

Values lie in [0, 1] (transition probabilities).

## Spectrograms & scalograms

Time-frequency representations for non-stationary series.

```python
from polars_ts import to_spectrogram, to_scalogram

# STFT spectrogram
specs = to_spectrogram(df, nperseg=64, noverlap=32, window="hann", log_scale=True)

# CWT scalogram (Morlet or Ricker wavelet)
scals = to_scalogram(df, wavelet="morlet", n_scales=32)
scals_custom = to_scalogram(df, wavelet="ricker", scales=np.geomspace(1, 50, 64))
```

## Vision model embeddings

Pass any of the above images through a pretrained vision model to get dense feature vectors.

```python
from polars_ts import to_gasf, extract_vision_embeddings

images = to_gasf(df)
embeddings = extract_vision_embeddings(images, model="resnet18")
# → DataFrame[unique_id, emb_0, emb_1, ..., emb_511]
```

Supported models: `resnet18`, `resnet50`, `vit_b_16`, `clip`.

## Image → clustering pipeline

```python
import numpy as np
from sklearn.cluster import KMeans
from polars_ts import to_gasf

images = to_gasf(df)
X = np.stack([img.flatten() for img in images.values()])
labels = KMeans(n_clusters=3, n_init=10).fit_predict(X)
```

Or use vision model embeddings for higher-quality features:

```python
from polars_ts import to_gasf, extract_vision_embeddings

images = to_gasf(df)
embeddings = extract_vision_embeddings(images, model="resnet18")
X = embeddings.drop("unique_id").to_numpy()
labels = KMeans(n_clusters=3, n_init=10).fit_predict(X)
```

## Further reading

- **Notebook 07**: [Time series similarity & clustering](https://github.com/drumtorben/polars-ts/blob/main/notebooks/07_time_series_similarity_clustering.ipynb)
- Wang & Oates (2015). *Encoding Time Series as Images for Classification*. AAAI.
- Hatami et al. (2018). *Classification of TS by CNNs Applied to Recurrence Plots*.
- Eckmann et al. (1987). *Recurrence Plots of Dynamical Systems*. Europhysics Letters.
