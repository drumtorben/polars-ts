"""Native deep learning forecasters (N-BEATS, PatchTST)."""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "NBEATSForecaster": ("polars_ts.dl.nbeats", "NBEATSForecaster"),
    "PatchTSTForecaster": ("polars_ts.dl.patchtst", "PatchTSTForecaster"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
