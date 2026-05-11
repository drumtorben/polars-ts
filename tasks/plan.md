# Tech Debt & Simplification Plan

## Executive Summary

The polars-ts codebase has grown rapidly with 80+ Python modules and 30+ Rust source files.
This plan covers: **duplicated code** (distance imports, pipeline helpers, `_extract_series`),
**inconsistent module patterns** (lazy vs eager imports, stdlib vs numpy random),
**large monolithic files**, **security hardening cleanups**, and **public API gaps**.

Phases 1-2 are complete. Phases 3-8 remain, ordered by effort/value.

**Dependencies:** Phases 6-8 depend on resolution of #247 (security hardening) and #214.

## Dependency Graph

```
polars_ts/__init__.py
  -> polars_ts_rs (Rust FFI)
  -> polars_ts._distance_dispatch (canonical Rust imports)
  -> polars_ts.pipeline -> polars_ts.models._time_utils (shared)
  -> polars_ts.global_model -> polars_ts.pipeline (shared helpers)
  -> polars_ts.registry -> joblib (model persistence)
  -> polars_ts.datasets (dataset loader with SHA-256)

clustering/* -> _distance_dispatch.py
clustering/{kmeans,kmedoids,scalable} -> stdlib random (INCONSISTENT)
adapters/embeddings.py -> trust_remote_code (Chronos only, missing from MOMENT)
```

## Phases

### Phase 1 — Centralize Distance Imports ✅ COMPLETE

### Phase 2 — Extract Shared Pipeline Helpers ✅ COMPLETE

---

### Phase 3 — Standardize Lazy Imports (LOW EFFORT / MEDIUM VALUE)

**Goal:** All submodule `__init__.py` files use `_lazy.py:make_getattr`.

**Status:** T3.1 and T3.2 are already done (confirmed by scan). Remaining:

**Tasks:**
- T3.1: ✅ `streaming/__init__.py` already uses `make_getattr`
- T3.2: ✅ Bayesian names already in `_LAZY_IMPORTS`, no special-case block
- T3.3: Convert `bayesian_var/__init__.py` from eager imports to `make_getattr`
- T3.4: Convert `models/bayesian_ets/__init__.py` from eager imports to `make_getattr`
- T3.5: Document `metrics/__init__.py` Polars namespace rationale
- T3.6: Run `test_lazy_imports.py`

**Acceptance:** All subpackage `__init__.py` files use `make_getattr`; no eager import blocks.

---

### Phase 4 — Split Large Files (MEDIUM EFFORT / MEDIUM VALUE)

**Goal:** No source file exceeds ~500 lines.

**Priority targets:**

| File | Lines | Split Strategy |
|------|-------|----------------|
| `reconciliation.py` | 590 | Split into `reconciliation/{bottom_up,top_down,mintrace,middle_out}.py` |
| `causal/synthetic_control.py` | 540 | Split model + inference |
| `bayesian/gp.py` | 539 | Extract kernels to `bayesian/gp_kernels.py` |
| `dl/multivariate.py` | 523 | Split `_MVPatchTST` and `iTransformer` into separate files |
| `pipeline.py` | 462 | Monitor (shared helpers already extracted) |
| `models/multistep.py` | 450 | Split `RecursiveForecaster` and `DirectForecaster` |
| `adapters/foundation_forecast.py` | 408 | One class per file |

**Note:** `bayesian_var.py` and `bayesian/mcmc.py` were already split in prior work.

**Tasks:**
- T4.1: Split `reconciliation.py` (590L) into subpackage
- T4.2: Split `bayesian/gp.py` — extract kernels
- T4.3: Split `dl/multivariate.py` — two unrelated models
- T4.4: Split `causal/synthetic_control.py`
- T4.5: Update all imports and re-exports
- T4.6: Run full test suite after each split

**Acceptance:** No source file exceeds 500 lines; all tests pass.

---

### Phase 5 — Deduplicate Shared Patterns (LOW EFFORT / HIGH VALUE)

**Goal:** Eliminate cross-module code duplication discovered by simplification scan.

**Tasks:**

- T5.1: **Extract `_extract_series`** — 3 near-identical copies in `adapters/embeddings.py:16`, `clustering/shapelets.py:21`, `features/rocket.py:22`. Create `polars_ts/_array_utils.py` with canonical version.

- T5.2: **Extract `_validate_horizon`** — 14 copies of `raise ValueError("Horizon h must be a positive integer")` across 8+ files. Create helper in `models/_time_utils.py`.

- T5.3: **Extract `_forecast_schema`** — schema dict `{id_col: ..., time_col: ..., "y_hat": pl.Float64}` repeated 9+ times in baselines/exponential_smoothing/multistep. Create helper in `models/_time_utils.py`.

**Acceptance:** `grep -r "_extract_series" polars_ts/ | wc -l` returns 1; horizon validation and schema construction are single-source.

---

### Phase 6 — Standardize Randomness (LOW EFFORT / MEDIUM VALUE)

**Goal:** All modules use `numpy.random.default_rng()` consistently.

**Depends on:** #247 merged (security hardening PR contains kaboudan.py fix).

**Remaining files using `stdlib random`:**

| File | Lines | Pattern |
|------|-------|---------|
| `clustering/kmeans.py` | 10, 100, 168 | `random.Random(seed).sample()`, `random.Random(seed+ci).randint()` |
| `clustering/kmedoids.py` | 9, 155 | `random.Random(seed).sample()` |
| `clustering/scalable.py` | 17, 93, 217, 224 | `random.Random(seed).sample()`, `.randint()`, `.choice()` |

**Files using legacy `np.random.RandomState`:**

| File | Lines |
|------|-------|
| `clustering/deep_cluster.py` | 216 |
| `clustering/contrastive.py` | 172 |

**Hard-coded seeds (not user-controllable):**

| File | Line | Issue |
|------|------|-------|
| `clustering/kshape.py` | 49 | `np.random.default_rng(42)` — no seed param |
| `probabilistic/conformal.py` | 265 | `np.random.default_rng(42)` — hard-coded |

**Tasks:**
- T6.1: Migrate `kmeans.py`, `kmedoids.py`, `scalable.py` from `stdlib random` to `np.random.default_rng`
- T6.2: Migrate `deep_cluster.py`, `contrastive.py` from `RandomState` to `default_rng`
- T6.3: Expose seed parameter in `kshape.py` and `conformal.py`
- T6.4: Run clustering + probabilistic tests

**Acceptance:** `grep -r "import random" polars_ts/` returns 0; no `np.random.RandomState` usage.

---

### Phase 7 — Security Hardening Cleanups (LOW EFFORT / MEDIUM VALUE)

**Goal:** Clean up issues found during the security review code quality scan.

**Depends on:** #247 merged.

**Tasks:**

- T7.1: **Remove dead code in `registry.py:71-72`** — `version = version or ...` is unreachable after the if/else block; `model_dir` reassignment is redundant.

- T7.2: **Cache `base.resolve()` in `_validate_path`** — currently calls `Path.resolve()` twice per validation (target + base comparison).

- T7.3: **Stream hash verification in `datasets.py`** — `read_bytes()` loads entire file (85MB for m5_y) into memory. Use chunked `hashlib.update()`.

- T7.4: **Atomic downloads in `datasets.py`** — download to `.tmp` then rename, so interrupted downloads don't leave partial files.

- T7.5: **Add `trust_remote_code` to `to_moment_embeddings()`** — inconsistent with `to_chronos_embeddings()` which already has this parameter.

- T7.6: Run registry + dataset + embedding tests.

**Acceptance:** No dead code in registry; hash streaming works for 85MB files; downloads are atomic.

---

### Phase 8 — Public API & Export Consistency (LOW EFFORT / HIGH VALUE)

**Goal:** Consistent public API surface — all public classes accessible from root `polars_ts`.

**Tasks:**

- T8.1: **Remove private functions from `__all__`** in `bayesian_var/__init__.py` (`_build_var_matrices`, `_estimate_sigma_from_ar`, `_minnesota_prior_precision`) and `models/bayesian_ets/__init__.py` (9 private `_` functions).

- T8.2: **Add missing entries to root `_LAZY_IMPORTS`:**
  - `MCMCResult` (in `bayesian/__init__` but missing from root)
  - `BayesianETSResult`
  - `NBEATSForecaster`, `PatchTSTForecaster`
  - `InceptionTimeClassifier`, `ResNetClassifier`, `RocketClassifier`, `MiniRocketClassifier`
  - `ParticleFilter`, `BayesianAnomalyDetector`
  - `Kaboudan`

- T8.3: **Audit `__all__` exports** across all modules to match actual public API.

- T8.4: Run `test_lazy_imports.py` + full import test.

**Acceptance:** All public classes importable from `polars_ts` directly; no private functions in `__all__`.

---

## Checkpoints

| After Phase | Verify |
|-------------|--------|
| Phase 3 | `pytest tests/test_lazy_imports.py` passes; all `__init__.py` use `make_getattr` |
| Phase 4 | Full `pytest` passes; `wc -l` confirms no file >500 lines |
| Phase 5 | Grep confirms single-source for `_extract_series`, horizon validation, schema construction |
| Phase 6 | No `import random` or `RandomState` in source; clustering tests pass |
| Phase 7 | No dead code; streaming hash; atomic downloads; tests pass |
| Phase 8 | All public classes in root `_LAZY_IMPORTS`; no private `__all__` entries |

## Risk Assessment

- **Phase 3:** Low risk — import pattern change with test coverage
- **Phase 4:** Medium risk — file splits may break notebooks/downstream
- **Phase 5:** Low risk — extracting shared code with tests
- **Phase 6:** Low risk — stdlib→numpy migration, well-tested
- **Phase 7:** Low risk — cleanup of recently added code
- **Phase 8:** Low risk — additive changes to import registry
