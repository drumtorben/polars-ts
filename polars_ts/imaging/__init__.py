from typing import Any


def __getattr__(name: str) -> Any:
    if name == "to_recurrence_plot":
        from polars_ts.imaging.recurrence import to_recurrence_plot

        return to_recurrence_plot
    if name == "rqa_features":
        from polars_ts.imaging.recurrence import rqa_features

        return rqa_features
    if name == "to_gasf":
        from polars_ts.imaging.angular import to_gasf

        return to_gasf
    if name == "to_gadf":
        from polars_ts.imaging.angular import to_gadf

        return to_gadf
    if name == "to_mtf":
        from polars_ts.imaging.transition import to_mtf

        return to_mtf
    if name == "to_spectrogram":
        from polars_ts.imaging.spectral import to_spectrogram

        return to_spectrogram
    if name == "to_scalogram":
        from polars_ts.imaging.spectral import to_scalogram

        return to_scalogram
    if name == "signature_features":
        from polars_ts.imaging.signature import signature_features

        return signature_features
    if name == "to_signature_image":
        from polars_ts.imaging.signature import to_signature_image

        return to_signature_image
    if name == "extract_vision_embeddings":
        from polars_ts.imaging.embeddings import extract_vision_embeddings

        return extract_vision_embeddings
    raise AttributeError(f"module 'polars_ts.imaging' has no attribute {name!r}")


__all__ = [
    "to_recurrence_plot",
    "rqa_features",
    "to_gasf",
    "to_gadf",
    "to_mtf",
    "to_spectrogram",
    "to_scalogram",
    "signature_features",
    "to_signature_image",
    "extract_vision_embeddings",
]
