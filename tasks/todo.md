# Tech Debt & Simplification — Task List

## Phase 1: Centralize Distance Imports ✅
- [x] T1.1–T1.5: Complete

## Phase 2: Extract Shared Pipeline Helpers ✅
- [x] T2.1–T2.6: Complete

## Phase 3: Standardize Lazy Imports
- [x] T3.1: `streaming/__init__.py` already uses `make_getattr`
- [x] T3.2: Bayesian names already in `_LAZY_IMPORTS`
- [ ] T3.3: Convert `bayesian_var/__init__.py` from eager to `make_getattr`
- [ ] T3.4: Convert `models/bayesian_ets/__init__.py` from eager to `make_getattr`
- [ ] T3.5: Document `metrics/__init__.py` Polars namespace rationale
- [ ] T3.6: Run `test_lazy_imports.py`

## Phase 4: Split Large Files
- [ ] T4.1: Split `reconciliation.py` (590L)
- [ ] T4.2: Split `bayesian/gp.py` (539L) — extract kernels
- [ ] T4.3: Split `dl/multivariate.py` (523L) — PatchTST vs iTransformer
- [ ] T4.4: Split `causal/synthetic_control.py` (540L)
- [ ] T4.5: Update all imports and re-exports
- [ ] T4.6: Run full test suite

## Phase 5: Deduplicate Shared Patterns
- [ ] T5.1: Extract `_extract_series` to `_array_utils.py` (3 copies → 1)
- [ ] T5.2: Extract `_validate_horizon` to `models/_time_utils.py` (14 copies → 1)
- [ ] T5.3: Extract `_forecast_schema` to `models/_time_utils.py` (9+ copies → 1)

## Phase 6: Standardize Randomness (depends on #247)
- [ ] T6.1: Migrate `kmeans.py`, `kmedoids.py`, `scalable.py` from stdlib `random` to `np.random.default_rng`
- [ ] T6.2: Migrate `deep_cluster.py`, `contrastive.py` from `RandomState` to `default_rng`
- [ ] T6.3: Expose seed parameter in `kshape.py:49` and `conformal.py:265`
- [ ] T6.4: Run clustering + probabilistic tests

## Phase 7: Security Hardening Cleanups (depends on #247)
- [ ] T7.1: Remove dead code `registry.py:71-72`
- [ ] T7.2: Cache `base.resolve()` in `_validate_path`
- [ ] T7.3: Stream hash verification in `datasets.py` (85MB file)
- [ ] T7.4: Atomic downloads in `datasets.py` (temp file + rename)
- [ ] T7.5: Add `trust_remote_code` param to `to_moment_embeddings()`
- [ ] T7.6: Run registry + dataset + embedding tests

## Phase 8: Public API & Export Consistency
- [ ] T8.1: Remove private `_` functions from `__all__` in `bayesian_var/` and `bayesian_ets/`
- [ ] T8.2: Add missing classes to root `_LAZY_IMPORTS` (MCMCResult, Kaboudan, DL forecasters, etc.)
- [ ] T8.3: Audit `__all__` exports across all modules
- [ ] T8.4: Run `test_lazy_imports.py`
