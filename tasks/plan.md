# Tech Debt Refactoring Plan

## Executive Summary

The polars-ts codebase has grown rapidly with 80+ Python modules and 30+ Rust source files.
The main tech debt categories are: **duplicated code** (distance imports, pipeline helpers),
**inconsistent module patterns** (lazy vs eager imports), and **large monolithic files**.

## Dependency Graph (relevant components)

```
polars_ts/__init__.py
  -> polars_ts_rs (Rust FFI)
  -> polars_ts.distance          (also imports polars_ts_rs)
  -> polars_ts._distance_dispatch (also imports polars_ts_rs)
  -> polars_ts.pipeline
       -> polars_ts.models.baselines (_infer_freq, _make_future_dates)
       -> polars_ts.models.multistep (Estimator protocol)
  -> polars_ts.global_model
       -> polars_ts.pipeline (shared helpers)
       -> polars_ts.models.baselines (_infer_freq, _make_future_dates)
  -> polars_ts.models.arima
       -> own _infer_freq / _make_future_dates (DUPLICATE)

clustering/classification modules -> _distance_dispatch.py
```

## Phases

### Phase 1 — Centralize Distance Imports (LOW EFFORT / HIGH VALUE)

**Goal:** Single source of truth for Rust FFI distance functions.

**Why:** 4 files (`__init__.py`, `distance.py`, `_distance_dispatch.py`, `clustering/kasba.py`)
all independently import the same 12 Rust functions. Adding a new distance metric requires
touching all 4 files.

**Tasks:**
- T1.1: Make `_distance_dispatch.py` the canonical import location
- T1.2: Update `distance.py` to import from `_distance_dispatch` instead of `polars_ts_rs`
- T1.3: Update `__init__.py` to re-export from `distance.py` (already partly done)
- T1.4: Update `clustering/kasba.py` to import from `_distance_dispatch`
- T1.5: Run tests to verify no regressions

**Acceptance:** `rg "from polars_ts_rs" polars_ts/` shows only `_distance_dispatch.py` and `__init__.py` (for PLUGIN_PATH).

### Phase 2 — Extract Shared Pipeline Helpers (LOW EFFORT / MEDIUM VALUE)

**Goal:** Eliminate duplicate `_inverse_single`, `_transform_buffer`, `_infer_freq`, `_make_future_dates`.

**Tasks:**
- T2.1: Extract `_inverse_single` and `_transform_buffer` to `polars_ts/transforms/_inverse.py`
- T2.2: Update `pipeline.py` and `global_model.py` to import from `_inverse.py`
- T2.3: Consolidate `_infer_freq` and `_make_future_dates` into `polars_ts/models/_time_utils.py`
- T2.4: Update `baselines.py`, `arima.py`, `pipeline.py`, `global_model.py` to import from `_time_utils.py`
- T2.5: Decide on `_infer_freq` behavior: `median` (baselines) vs `mode` (arima) — pick one or make configurable
- T2.6: Run full test suite

**Acceptance:** No duplicated `_inverse_single`, `_transform_buffer`, `_infer_freq`, or `_make_future_dates` across codebase.

### Phase 3 — Standardize Lazy Import Pattern (LOW EFFORT / MEDIUM VALUE)

**Goal:** All submodule `__init__.py` files use `_lazy.py:make_getattr`.

**Tasks:**
- T3.1: Convert `streaming/__init__.py` from eager imports to `make_getattr` pattern
- T3.2: Move bayesian special-case from `__init__.py` into `_LAZY_IMPORTS` dict
- T3.3: Audit `metrics/__init__.py` — keep Metrics class (it's a Polars namespace, not lazy-importable) but document the reason
- T3.4: Run `test_lazy_imports.py` to verify all lazy imports still resolve

**Acceptance:** `streaming/__init__.py` uses `make_getattr`; bayesian names are in `_LAZY_IMPORTS`; no special-case `if` blocks in `__init__.py`.

---

### Phase 4 — Split Large Files (MEDIUM EFFORT / MEDIUM VALUE)

**Goal:** No file exceeds ~500 lines; complex modules are split by responsibility.

**Priority targets (by size and complexity):**

| File | Lines | Split Strategy |
|------|-------|----------------|
| `bayesian_var.py` | 892 | Split into `bayesian_var/{model,priors,results}.py` |
| `models/bayesian_ets.py` | 854 | Split into `{model,gibbs_sampler,priors}.py` |
| `bayesian/mcmc.py` | 691 | Split into `{samplers,diagnostics,utils}.py` |
| `causal/causal_impact.py` | 648 | Split into `{model,inference,results}.py` |

**Tasks:**
- T4.1: Split `bayesian_var.py` into subpackage
- T4.2: Split `models/bayesian_ets.py` into subpackage
- T4.3: Split `bayesian/mcmc.py` into focused modules
- T4.4: Split `causal/causal_impact.py`
- T4.5: Update all imports and re-exports
- T4.6: Run full test suite after each split

**Acceptance:** No source file exceeds 500 lines; all tests pass.

---

### Phase 5 — Minor Cleanups (LOW EFFORT / LOW VALUE)

**Tasks:**
- T5.1: Ensure `_distance_dispatch.py` has a test (currently only tested indirectly)
- T5.2: Add `py.typed` marker check to CI (already exists as file)
- T5.3: Audit `__all__` exports match actual public API across all modules

**Acceptance:** Direct test for `_distance_dispatch.py`; CI validates typed marker.

---

## Checkpoints

| After Phase | Verify |
|-------------|--------|
| Phase 1 | `pytest tests/distance/` passes; single import location |
| Phase 2 | `pytest tests/models/ tests/test_pipeline.py tests/test_global_model.py` passes |
| Phase 3 | `pytest tests/test_lazy_imports.py` passes; grep shows no eager streaming imports |
| Phase 4 | Full `pytest` passes; `wc -l` confirms no file >500 lines |
| Phase 5 | Full CI green |

## Risk Assessment

- **Phase 1-3:** Low risk — import reorganization with full test coverage
- **Phase 4:** Medium risk — file splits may break imports in notebooks/downstream code
- **Phase 5:** Negligible risk
