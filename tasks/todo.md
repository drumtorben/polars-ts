# Tech Debt Refactoring — Task List

## Phase 1: Centralize Distance Imports
- [x] T1.1: Make `_distance_dispatch.py` the canonical Rust FFI import location
- [x] T1.2: Update `distance.py` to import from `_distance_dispatch`
- [x] T1.3: Update `__init__.py` to import from `_distance_dispatch`
- [x] T1.4: N/A — `clustering/kasba.py` imports `kasba_fit`/`kasba_predict` (not distance functions)
- [x] T1.5: Run distance tests — 318/318 passed

## Phase 2: Extract Shared Pipeline Helpers
- [ ] T2.1: Extract `_inverse_single` + `_transform_buffer` to `transforms/_inverse.py`
- [ ] T2.2: Update `pipeline.py` and `global_model.py` imports
- [ ] T2.3: Consolidate `_infer_freq` + `_make_future_dates` into `models/_time_utils.py`
- [ ] T2.4: Update `baselines.py`, `arima.py`, `pipeline.py`, `global_model.py`
- [ ] T2.5: Resolve `_infer_freq` behavior divergence (median vs mode)
- [ ] T2.6: Run full test suite

## Phase 3: Standardize Lazy Imports
- [ ] T3.1: Convert `streaming/__init__.py` to `make_getattr` pattern
- [ ] T3.2: Move bayesian names from special-case block into `_LAZY_IMPORTS`
- [ ] T3.3: Document `metrics/__init__.py` Polars namespace rationale
- [ ] T3.4: Run `test_lazy_imports.py`

## Phase 4: Split Large Files
- [ ] T4.1: Split `bayesian_var.py` (892 lines) into subpackage
- [ ] T4.2: Split `models/bayesian_ets.py` (854 lines) into subpackage
- [ ] T4.3: Split `bayesian/mcmc.py` (691 lines)
- [ ] T4.4: Split `causal/causal_impact.py` (648 lines)
- [ ] T4.5: Update imports and re-exports
- [ ] T4.6: Run full test suite

## Phase 5: Minor Cleanups
- [x] T5.1: Add direct test for `_distance_dispatch.py` (done in Phase 1: `test_import_centralization.py`)
- [ ] T5.2: Audit `__all__` exports across all modules
