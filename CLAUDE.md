## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current

## Project Architecture

- **Hybrid Python/Rust codebase**: Python package at `polars_ts/`, Rust FFI via `polars_ts_rs` (built with maturin) at `src/`
- **Lazy imports**: All submodule `__init__.py` files should use `polars_ts._lazy.make_getattr` pattern
- **Distance dispatch**: `_distance_dispatch.py` is the canonical location for Rust distance function imports; other modules should import from there, not directly from `polars_ts_rs`
- **Shared helpers**: `pipeline.py` owns `_build_feature_df`, `_apply_transform`, `_build_step_features`; `global_model.py` imports from there
- **Time utils**: `_infer_freq` and `_make_future_dates` live in `models/baselines.py` (canonical) — `models/arima.py` has its own copy (tech debt)

## Conventions

- **Python**: ruff for linting/formatting (line-length=120, target=py312). Always run `ruff check --fix` + `ruff format` before committing.
- **Tests**: pytest with coverage. Test files mirror source structure under `tests/`.
- **Rust**: Cargo workspace with pyo3 bindings. `cargo test` for Rust tests.
- **CI**: GitHub Actions v4 (never use deprecated v3 actions).
- **Commits**: Use conventional commit style. Never skip hooks (`--no-verify`).

## Key Files

- `polars_ts/__init__.py` — Central lazy-import registry (`_LAZY_IMPORTS` dict)
- `polars_ts/_lazy.py` — `make_getattr` factory for submodule `__init__.py`
- `polars_ts/_distance_dispatch.py` — Rust FFI distance function registry
- `polars_ts/pipeline.py` — ForecastPipeline + shared feature engineering helpers
- `polars_ts/global_model.py` — GlobalForecaster (extends pipeline pattern)
- `polars_ts/metrics/__init__.py` — Polars DataFrame namespace (`df.pts.mae()`)
- `tasks/plan.md` — Tech debt refactoring plan
- `tasks/todo.md` — Task checklist

## Tech Debt Hotspots

See `tasks/plan.md` for the full plan. Key issues:
1. **Duplicate Rust imports** — 4 files import same 12 functions from `polars_ts_rs`
2. **Duplicate helpers** — `_inverse_single`, `_transform_buffer` duplicated in pipeline.py and global_model.py
3. **Duplicate `_infer_freq`** — baselines.py (median) vs arima.py (mode)
4. **Inconsistent lazy imports** — `streaming/__init__.py` uses eager imports; bayesian has special-case block
5. **Large files** — `bayesian_var.py` (892L), `bayesian_ets.py` (854L), `mcmc.py` (691L)
