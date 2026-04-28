from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "to_recurrence_plot": ("polars_ts.imaging.recurrence", "to_recurrence_plot"),
    "rqa_features": ("polars_ts.imaging.recurrence", "rqa_features"),
    "to_gasf": ("polars_ts.imaging.angular", "to_gasf"),
    "to_gadf": ("polars_ts.imaging.angular", "to_gadf"),
    "to_mtf": ("polars_ts.imaging.transition", "to_mtf"),
    "to_spectrogram": ("polars_ts.imaging.spectral", "to_spectrogram"),
    "to_scalogram": ("polars_ts.imaging.spectral", "to_scalogram"),
    "signature_features": ("polars_ts.imaging.signature", "signature_features"),
    "to_signature_image": ("polars_ts.imaging.signature", "to_signature_image"),
    "extract_vision_embeddings": ("polars_ts.imaging.embeddings", "extract_vision_embeddings"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
