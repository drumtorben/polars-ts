from typing import Any


def __getattr__(name: str) -> Any:
    if name == "SCUM":
        try:
            from polars_ts.models.scum import SCUM
        except ImportError:
            raise ImportError(
                "statsforecast is required for SCUM. " "Install it with: pip install polars-timeseries[forecast]"
            ) from None
        return SCUM
    raise AttributeError(f"module 'polars_ts.models' has no attribute {name!r}")


__all__ = ["SCUM"]
