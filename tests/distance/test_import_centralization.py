"""Tests that distance Rust FFI imports are centralized in _distance_dispatch.py.

Phase 1 of tech debt refactoring (#215): after the refactor, distance.py and
__init__.py should import compute_pairwise_* from _distance_dispatch, NOT
directly from polars_ts_rs.
"""

import ast
from pathlib import Path

import pytest

import polars_ts
from polars_ts._distance_dispatch import (
    _DISTANCE_FUNCS,
    compute_distances,
    pairwise_to_dict,
)

POLARS_TS_ROOT = Path(polars_ts.__file__).parent


class TestImportCentralization:
    """Verify that only _distance_dispatch.py imports from polars_ts_rs.polars_ts_rs."""

    def _get_imports_from_polars_ts_rs(self, filepath: Path) -> list[str]:
        """Parse a Python file and return names imported from polars_ts_rs.polars_ts_rs."""
        source = filepath.read_text()
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "polars_ts_rs.polars_ts_rs" and node.names:
                    for alias in node.names:
                        if alias.name.startswith("compute_pairwise_"):
                            imported.append(alias.name)
        return imported

    def test_distance_dispatch_is_canonical_source(self):
        """_distance_dispatch.py should import all 12 distance functions from Rust."""
        dispatch = POLARS_TS_ROOT / "_distance_dispatch.py"
        imports = self._get_imports_from_polars_ts_rs(dispatch)
        assert len(imports) == 12
        expected = {
            "compute_pairwise_ddtw",
            "compute_pairwise_dtw",
            "compute_pairwise_dtw_multi",
            "compute_pairwise_edr",
            "compute_pairwise_erp",
            "compute_pairwise_frechet",
            "compute_pairwise_lcss",
            "compute_pairwise_msm",
            "compute_pairwise_msm_multi",
            "compute_pairwise_sbd",
            "compute_pairwise_twe",
            "compute_pairwise_wdtw",
        }
        assert set(imports) == expected

    def test_distance_py_does_not_import_from_rust(self):
        """distance.py should NOT import compute_pairwise_* directly from polars_ts_rs."""
        distance = POLARS_TS_ROOT / "distance.py"
        imports = self._get_imports_from_polars_ts_rs(distance)
        assert imports == [], (
            f"distance.py should import from _distance_dispatch, not polars_ts_rs. " f"Found direct imports: {imports}"
        )

    def test_init_py_does_not_import_distance_funcs_from_rust(self):
        """__init__.py should NOT import compute_pairwise_* directly from polars_ts_rs."""
        init = POLARS_TS_ROOT / "__init__.py"
        imports = self._get_imports_from_polars_ts_rs(init)
        assert imports == [], (
            f"__init__.py should import from _distance_dispatch, not polars_ts_rs. " f"Found direct imports: {imports}"
        )


class TestDistanceDispatchRegistry:
    """Verify _distance_dispatch.py provides a complete, usable registry."""

    def test_all_methods_in_registry(self):
        """All 12 distance methods are in the _DISTANCE_FUNCS registry."""
        expected_methods = {
            "dtw",
            "ddtw",
            "wdtw",
            "msm",
            "erp",
            "lcss",
            "twe",
            "sbd",
            "frechet",
            "edr",
            "dtw_multi",
            "msm_multi",
        }
        assert set(_DISTANCE_FUNCS.keys()) == expected_methods

    def test_all_registry_values_are_callable(self):
        """Every entry in _DISTANCE_FUNCS is callable."""
        for name, func in _DISTANCE_FUNCS.items():
            assert callable(func), f"{name} is not callable"

    def test_compute_distances_dispatches_correctly(self, two_series):
        """compute_distances produces the same result as direct Rust call."""
        result = compute_distances(two_series, two_series, method="dtw")
        assert "dtw" in result.columns
        assert len(result) > 0

    def test_compute_distances_unknown_method(self, two_series):
        with pytest.raises(ValueError, match="Unknown distance method"):
            compute_distances(two_series, two_series, method="invalid")

    def test_compute_distances_unexpected_kwarg(self, two_series):
        with pytest.raises(ValueError, match="Unexpected kwargs"):
            compute_distances(two_series, two_series, method="ddtw", g=0.5)


class TestPublicAPIPreserved:
    """Verify the public API still works after import centralization."""

    def test_top_level_compute_pairwise_dtw(self, two_series):
        """polars_ts.compute_pairwise_dtw still works."""
        result = polars_ts.compute_pairwise_dtw(two_series, two_series)
        assert "dtw" in result.columns

    def test_top_level_compute_pairwise_distance(self, two_series):
        """polars_ts.compute_pairwise_distance still works."""
        result = polars_ts.compute_pairwise_distance(two_series, two_series)
        assert "dtw" in result.columns

    def test_all_distance_names_in__all__(self):
        """All 12 compute_pairwise_* names are in polars_ts.__all__."""
        expected = {
            "compute_pairwise_ddtw",
            "compute_pairwise_dtw",
            "compute_pairwise_dtw_multi",
            "compute_pairwise_edr",
            "compute_pairwise_erp",
            "compute_pairwise_frechet",
            "compute_pairwise_lcss",
            "compute_pairwise_msm",
            "compute_pairwise_msm_multi",
            "compute_pairwise_sbd",
            "compute_pairwise_twe",
            "compute_pairwise_wdtw",
        }
        assert expected.issubset(set(polars_ts.__all__))

    @pytest.mark.parametrize(
        "name",
        [
            "compute_pairwise_dtw",
            "compute_pairwise_ddtw",
            "compute_pairwise_wdtw",
            "compute_pairwise_msm",
            "compute_pairwise_erp",
            "compute_pairwise_lcss",
            "compute_pairwise_twe",
            "compute_pairwise_sbd",
            "compute_pairwise_frechet",
            "compute_pairwise_edr",
            "compute_pairwise_dtw_multi",
            "compute_pairwise_msm_multi",
        ],
    )
    def test_each_function_accessible_from_top_level(self, name):
        """Each compute_pairwise_* function is accessible from polars_ts."""
        func = getattr(polars_ts, name)
        assert callable(func)

    def test_pairwise_to_dict_accessible(self):
        """pairwise_to_dict is importable from _distance_dispatch."""
        assert callable(pairwise_to_dict)
