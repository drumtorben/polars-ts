"""Shared lazy-import helper for polars-ts modules.

Provides a generic ``make_getattr`` factory that turns a flat registry
dict into a module-level ``__getattr__`` function.  This eliminates
the duplicated if-chain boilerplate across submodule ``__init__.py``
files.

Usage in a submodule ``__init__.py``::

    from polars_ts._lazy import make_getattr

    _IMPORTS: dict[str, tuple[str, str]] = {
        "KShape": ("polars_ts.clustering.kshape", "KShape"),
        "kmedoids": ("polars_ts.clustering.kmedoids", "kmedoids"),
    }

    __getattr__, __all__ = make_getattr(_IMPORTS, __name__)
"""

from __future__ import annotations

import importlib
from typing import Any


def make_getattr(
    registry: dict[str, tuple[str, str]],
    module_name: str,
) -> tuple[Any, list[str]]:
    """Create ``__getattr__`` and ``__all__`` from a lazy-import registry.

    Parameters
    ----------
    registry
        Mapping from public name to ``(module_path, attribute_name)``.
    module_name
        The ``__name__`` of the calling module (for error messages).

    Returns
    -------
    tuple
        ``(__getattr__, __all__)`` ready to be assigned at module level.

    """

    def __getattr__(name: str) -> Any:
        if name in registry:
            mod_path, attr = registry[name]
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__, list(registry.keys())
