import numpy as np
import polars as pl
import pytest

scipy_signal = pytest.importorskip("scipy.signal")

from polars_ts.imaging.spectral import to_scalogram, to_spectrogram  # noqa: E402


@pytest.fixture
def sample_data():
    """Two series with 200 points each."""
    t = np.linspace(0, 1, 200)
    # Series A: pure sine wave
    a = np.sin(2 * np.pi * 10 * t)
    # Series B: chirp (increasing frequency)
    b = np.sin(2 * np.pi * (5 * t + 10 * t**2))
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 200 + ["B"] * 200,
            "y": a.tolist() + b.tolist(),
        }
    )


class TestToSpectrogram:
    def test_output_keys(self, sample_data):
        images = to_spectrogram(sample_data, nperseg=32)
        assert set(images.keys()) == {"A", "B"}

    def test_output_shape(self, sample_data):
        images = to_spectrogram(sample_data, nperseg=32)
        for img in images.values():
            assert img.ndim == 2
            # freq axis = nperseg // 2 + 1
            assert img.shape[0] == 17

    def test_non_negative(self, sample_data):
        images = to_spectrogram(sample_data, nperseg=32)
        for img in images.values():
            assert img.min() >= 0

    def test_log_scale(self, sample_data):
        img_log = to_spectrogram(sample_data, nperseg=32, log_scale=True)
        img_raw = to_spectrogram(sample_data, nperseg=32, log_scale=False)
        # log1p values should be smaller than raw for values > ~1.7
        assert not np.allclose(img_log["A"], img_raw["A"])

    def test_nperseg_affects_shape(self, sample_data):
        img_32 = to_spectrogram(sample_data, nperseg=32)
        img_64 = to_spectrogram(sample_data, nperseg=64)
        assert img_32["A"].shape[0] != img_64["A"].shape[0]

    def test_noverlap_parameter(self, sample_data):
        img_default = to_spectrogram(sample_data, nperseg=32)
        img_custom = to_spectrogram(sample_data, nperseg=32, noverlap=16)
        # Default noverlap is nperseg // 2 = 16, so these should be equal
        np.testing.assert_allclose(img_default["A"], img_custom["A"])

    def test_window_parameter(self, sample_data):
        img_hann = to_spectrogram(sample_data, nperseg=32, window="hann")
        img_hamming = to_spectrogram(sample_data, nperseg=32, window="hamming")
        assert not np.allclose(img_hann["A"], img_hamming["A"])

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 100, "val": np.sin(np.linspace(0, 10, 100)).tolist()})
        images = to_spectrogram(df, nperseg=16, id_col="sid", target_col="val")
        assert "X" in images

    def test_single_series(self):
        df = pl.DataFrame({"unique_id": ["A"] * 100, "y": np.random.default_rng(0).random(100).tolist()})
        images = to_spectrogram(df, nperseg=16)
        assert len(images) == 1


class TestToScalogram:
    def test_output_keys(self, sample_data):
        images = to_scalogram(sample_data)
        assert set(images.keys()) == {"A", "B"}

    def test_output_shape(self, sample_data):
        images = to_scalogram(sample_data, n_scales=16)
        for img in images.values():
            assert img.ndim == 2
            assert img.shape[0] == 16  # n_scales
            assert img.shape[1] == 200  # series length

    def test_non_negative(self, sample_data):
        images = to_scalogram(sample_data)
        for img in images.values():
            assert img.min() >= 0

    def test_morlet_wavelet(self, sample_data):
        images = to_scalogram(sample_data, wavelet="morlet")
        assert len(images) == 2

    def test_ricker_wavelet(self, sample_data):
        images = to_scalogram(sample_data, wavelet="ricker")
        assert len(images) == 2

    def test_custom_scales(self, sample_data):
        scales = np.array([1, 2, 4, 8, 16, 32], dtype=np.float64)
        images = to_scalogram(sample_data, scales=scales)
        for img in images.values():
            assert img.shape[0] == 6

    def test_n_scales_parameter(self, sample_data):
        img_16 = to_scalogram(sample_data, n_scales=16)
        img_32 = to_scalogram(sample_data, n_scales=32)
        assert img_16["A"].shape[0] == 16
        assert img_32["A"].shape[0] == 32

    def test_unknown_wavelet_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unknown wavelet"):
            to_scalogram(sample_data, wavelet="nonexistent")

    def test_custom_columns(self):
        df = pl.DataFrame({"sid": ["X"] * 100, "val": np.sin(np.linspace(0, 10, 100)).tolist()})
        images = to_scalogram(df, id_col="sid", target_col="val")
        assert "X" in images

    def test_wavelet_aliases(self, sample_data):
        img_morlet = to_scalogram(sample_data, wavelet="morlet", n_scales=8)
        img_morl = to_scalogram(sample_data, wavelet="morl", n_scales=8)
        np.testing.assert_allclose(img_morlet["A"], img_morl["A"])
