from typing import Any

from polars_ts.changepoint.cusum import cusum


def __getattr__(name: str) -> Any:
    if name == "pelt":
        from polars_ts.changepoint.pelt import pelt

        return pelt
    if name == "bocpd":
        from polars_ts.changepoint.bocpd import bocpd

        return bocpd
    if name == "regime_detect":
        from polars_ts.changepoint.regime import regime_detect

        return regime_detect
    raise AttributeError(f"module 'polars_ts.changepoint' has no attribute {name!r}")


__all__ = ["cusum", "pelt", "bocpd", "regime_detect"]
