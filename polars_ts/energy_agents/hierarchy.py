"""Grid hierarchy for energy/demand forecasting agents.

Represents the region -> grid -> household topology and exposes it in the
child -> parent tree form consumed by :func:`polars_ts.reconciliation.reconcile`.
"""

from __future__ import annotations


class GridHierarchy:
    """Three-level energy grid topology: region -> grids -> households.

    Parameters
    ----------
    region
        Name of the top-level region node.
    structure
        Mapping ``grid_name -> [household_name, ...]``.

    """

    def __init__(self, region: str, structure: dict[str, list[str]]) -> None:
        if not structure:
            raise ValueError("structure must contain at least one grid")
        self.region = region
        self.structure = {g: list(hs) for g, hs in structure.items()}
        # Detect duplicate household ids across grids.
        seen: set[str] = set()
        for households in self.structure.values():
            for h in households:
                if h in seen:
                    raise ValueError(f"household {h!r} appears under multiple grids")
                seen.add(h)

    @property
    def grids(self) -> list[str]:
        """Grid (mid-level) node names."""
        return list(self.structure.keys())

    @property
    def households(self) -> list[str]:
        """Household (bottom-level) node names."""
        return [h for hs in self.structure.values() for h in hs]

    def all_nodes(self) -> list[str]:
        """All node names, ordered region -> grids -> households."""
        return [self.region, *self.grids, *self.households]

    def tree(self) -> dict[str, str]:
        """Return the child -> parent mapping for reconciliation.

        The top (region) node is intentionally omitted, matching the tree form
        expected by :func:`polars_ts.reconciliation.reconcile`.
        """
        mapping: dict[str, str] = {}
        for grid, households in self.structure.items():
            mapping[grid] = self.region
            for h in households:
                mapping[h] = grid
        return mapping

    def children(self, node: str) -> list[str]:
        """Direct children of ``node`` (empty for households)."""
        if node == self.region:
            return self.grids
        if node in self.structure:
            return list(self.structure[node])
        return []
