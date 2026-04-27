#!/usr/bin/env python3
"""Normalize Jupyter notebook JSON structure for consistent CI.

Ensures:
- Cell source is list-of-lines format (not a single string)
- All cells have ``metadata: {}``
- Code cells have ``execution_count: null`` and ``outputs: []``
- Consistent key ordering within cells

Run directly::

    python scripts/normalize_notebooks.py notebooks/*.ipynb

Or via pre-commit (see .pre-commit-config.yaml).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Canonical key order for cells
_CODE_KEYS = ("cell_type", "execution_count", "id", "metadata", "outputs", "source")
_MD_KEYS = ("cell_type", "id", "metadata", "source")


def _normalize_source(source: str | list[str]) -> list[str]:
    """Convert source to list-of-lines format with trailing newlines."""
    if isinstance(source, str):
        lines = source.split("\n")
    else:
        # Already a list — rejoin and re-split to normalize
        lines = "".join(source).split("\n")

    if not lines:
        return []

    # Each line gets a trailing newline except the last
    result = [line + "\n" for line in lines[:-1]]
    result.append(lines[-1])
    return result


def _normalize_cell(cell: dict) -> dict:
    """Normalize a single cell."""
    cell_type = cell.get("cell_type", "code")

    # Ensure required fields
    cell.setdefault("metadata", {})
    cell["source"] = _normalize_source(cell.get("source", []))

    if cell_type == "code":
        cell.setdefault("execution_count", None)
        cell.setdefault("outputs", [])
        # Clear outputs and execution count for clean diffs
        cell["execution_count"] = None
        cell["outputs"] = []

    # Reorder keys
    key_order = _CODE_KEYS if cell_type == "code" else _MD_KEYS
    ordered: dict = {}
    for key in key_order:
        if key in cell:
            ordered[key] = cell[key]
    # Preserve any extra keys not in the canonical order
    for key, val in cell.items():
        if key not in ordered:
            ordered[key] = val

    return ordered


def normalize_notebook(path: Path) -> bool:
    """Normalize a notebook file in-place. Returns True if modified."""
    content = path.read_text(encoding="utf-8")
    nb = json.loads(content)

    nb["cells"] = [_normalize_cell(cell) for cell in nb.get("cells", [])]

    normalized = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if normalized != content:
        path.write_text(normalized, encoding="utf-8")
        return True
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: normalize_notebooks.py <notebook> [<notebook> ...]", file=sys.stderr)
        return 1

    modified = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists() or path.suffix != ".ipynb":
            continue
        if normalize_notebook(path):
            modified.append(str(path))

    if modified:
        for p in modified:
            print(f"Normalized: {p}")
        return 1  # non-zero so pre-commit knows files were modified

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
