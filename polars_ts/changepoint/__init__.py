from typing import Any

from polars_ts._lazy import make_getattr
from polars_ts.changepoint.cusum import cusum  # eager — Rust plugin

_IMPORTS: dict[str, tuple[str, str]] = {
    "pelt": ("polars_ts.changepoint.pelt", "pelt"),
    "bocpd": ("polars_ts.changepoint.bocpd", "bocpd"),
    "regime_detect": ("polars_ts.changepoint.regime", "regime_detect"),
}

_getattr, _all = make_getattr(_IMPORTS, __name__)
__all__ = ["cusum", *_all]


def __getattr__(name: str) -> Any:
    return _getattr(name)
