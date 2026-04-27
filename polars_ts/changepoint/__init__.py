from polars_ts._lazy import make_getattr
from polars_ts.changepoint.cusum import cusum  # noqa: F401 — eager Rust plugin

_IMPORTS: dict[str, tuple[str, str]] = {
    "pelt": ("polars_ts.changepoint.pelt", "pelt"),
    "bocpd": ("polars_ts.changepoint.bocpd", "bocpd"),
    "regime_detect": ("polars_ts.changepoint.regime", "regime_detect"),
}

__getattr__, _all = make_getattr(_IMPORTS, __name__)
__all__ = ["cusum", *_all]
