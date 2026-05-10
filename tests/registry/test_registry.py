"""Tests for model registry and experiment tracking."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from polars_ts.registry.experiment import Experiment, Run
from polars_ts.registry.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyModel:
    """Minimal model with fit/predict for testing."""

    def __init__(self, bias: float = 0.0):
        self.bias = bias
        self.is_fitted_: bool = False

    def fit(self, _df: pl.DataFrame) -> _DummyModel:
        self.is_fitted_ = True
        return self

    def predict(self, _df: pl.DataFrame, *, h: int = 1) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "unique_id": ["A"] * h,
                "ds": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(h)],
                "y_hat": [self.bias] * h,
            }
        )


def _make_ts(n: int = 20) -> pl.DataFrame:
    base = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": [base + timedelta(days=i) for i in range(n)],
            "y": [float(i) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Experiment / Run dataclass tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_create_run(self):
        run = Run(
            run_id="run-001",
            model_name="dummy",
            config={"bias": 1.0},
            metrics={"mae": 0.5, "rmse": 0.7},
        )
        assert run.run_id == "run-001"
        assert run.model_name == "dummy"
        assert run.config == {"bias": 1.0}
        assert run.metrics["mae"] == 0.5
        assert isinstance(run.timestamp, str)

    def test_run_to_dict_roundtrip(self):
        run = Run(
            run_id="run-002",
            model_name="pipeline",
            config={"lags": [1, 2]},
            metrics={"mae": 1.2},
            tags={"env": "test"},
        )
        d = run.to_dict()
        restored = Run.from_dict(d)
        assert restored.run_id == run.run_id
        assert restored.model_name == run.model_name
        assert restored.config == run.config
        assert restored.metrics == run.metrics
        assert restored.tags == run.tags


class TestExperiment:
    def test_create_experiment(self):
        exp = Experiment(name="forecast-v1")
        assert exp.name == "forecast-v1"
        assert len(exp.runs) == 0

    def test_log_run(self):
        exp = Experiment(name="test-exp")
        run = exp.log_run(
            model_name="dummy",
            config={"bias": 0.0},
            metrics={"mae": 0.3},
        )
        assert len(exp.runs) == 1
        assert exp.runs[0].run_id == run.run_id
        assert run.model_name == "dummy"

    def test_log_multiple_runs(self):
        exp = Experiment(name="multi")
        exp.log_run(model_name="m1", config={}, metrics={"mae": 1.0})
        exp.log_run(model_name="m2", config={}, metrics={"mae": 0.5})
        assert len(exp.runs) == 2

    def test_best_run(self):
        exp = Experiment(name="best-test")
        exp.log_run(model_name="bad", config={}, metrics={"mae": 2.0, "rmse": 3.0})
        exp.log_run(model_name="good", config={}, metrics={"mae": 0.5, "rmse": 1.0})
        exp.log_run(model_name="mid", config={}, metrics={"mae": 1.0, "rmse": 2.0})

        best = exp.best_run("mae")
        assert best.model_name == "good"

        best_rmse = exp.best_run("rmse")
        assert best_rmse.model_name == "good"

    def test_best_run_higher_is_better(self):
        exp = Experiment(name="higher")
        exp.log_run(model_name="low", config={}, metrics={"r2": 0.7})
        exp.log_run(model_name="high", config={}, metrics={"r2": 0.95})

        best = exp.best_run("r2", higher_is_better=True)
        assert best.model_name == "high"

    def test_leaderboard(self):
        exp = Experiment(name="board")
        exp.log_run(model_name="m1", config={"a": 1}, metrics={"mae": 2.0, "rmse": 3.0})
        exp.log_run(model_name="m2", config={"a": 2}, metrics={"mae": 0.5, "rmse": 1.0})

        board = exp.leaderboard("mae")
        assert isinstance(board, pl.DataFrame)
        assert len(board) == 2
        assert board["model_name"][0] == "m2"  # best first
        assert "mae" in board.columns
        assert "run_id" in board.columns

    def test_to_dataframe(self):
        exp = Experiment(name="df-test")
        exp.log_run(model_name="m1", config={}, metrics={"mae": 1.0})
        exp.log_run(model_name="m2", config={}, metrics={"mae": 0.5})

        df = exp.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "run_id" in df.columns
        assert "model_name" in df.columns
        assert "mae" in df.columns

    def test_empty_experiment_best_run_raises(self):
        exp = Experiment(name="empty")
        with pytest.raises(ValueError, match="no runs"):
            exp.best_run("mae")


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_save_and_load_model(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        model = _DummyModel(bias=42.0)
        model.fit(_make_ts())

        registry.save_model(model, "my-model", version="v1")

        loaded = registry.load_model("my-model", version="v1")
        assert isinstance(loaded, _DummyModel)
        assert loaded.bias == 42.0
        assert loaded.is_fitted_

    def test_save_uses_joblib_format(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.save_model(_DummyModel(), "fmt-test", version="v1")
        assert (tmp_path / "models" / "fmt-test" / "v1" / "model.joblib").exists()
        assert not (tmp_path / "models" / "fmt-test" / "v1" / "model.pkl").exists()

    def test_load_legacy_pickle_fallback(self, tmp_path: Path):
        import pickle

        registry = ModelRegistry(tmp_path)
        # Simulate a legacy pickle file
        legacy_dir = tmp_path / "models" / "legacy" / "v1"
        legacy_dir.mkdir(parents=True)
        with open(legacy_dir / "model.pkl", "wb") as f:
            pickle.dump(_DummyModel(bias=99.0), f)

        loaded = registry.load_model("legacy", version="v1")
        assert isinstance(loaded, _DummyModel)
        assert loaded.bias == 99.0

    def test_save_model_auto_version(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        model = _DummyModel()

        v1 = registry.save_model(model, "auto-ver")
        v2 = registry.save_model(model, "auto-ver")

        assert v1 != v2
        # Both versions loadable
        registry.load_model("auto-ver", version=v1)
        registry.load_model("auto-ver", version=v2)

    def test_list_models(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.save_model(_DummyModel(), "alpha")
        registry.save_model(_DummyModel(), "beta")

        names = registry.list_models()
        assert set(names) == {"alpha", "beta"}

    def test_list_versions(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.save_model(_DummyModel(), "versioned")
        v2 = registry.save_model(_DummyModel(), "versioned")

        versions = registry.list_versions("versioned")
        assert v1 in versions
        assert v2 in versions
        assert len(versions) == 2

    def test_load_latest(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.save_model(_DummyModel(bias=1.0), "latest-test")
        registry.save_model(_DummyModel(bias=2.0), "latest-test")

        loaded = registry.load_model("latest-test")
        assert loaded.bias == 2.0

    def test_load_nonexistent_raises(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.load_model("does-not-exist")

    def test_save_with_metadata(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        model = _DummyModel(bias=5.0)
        model.fit(_make_ts())

        registry.save_model(
            model,
            "meta-model",
            version="v1",
            metadata={"dataset": "test", "horizon": 3},
        )

        meta = registry.get_metadata("meta-model", version="v1")
        assert meta["dataset"] == "test"
        assert meta["horizon"] == 3

    def test_delete_model(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        registry.save_model(_DummyModel(), "to-delete")
        assert "to-delete" in registry.list_models()

        registry.delete_model("to-delete")
        assert "to-delete" not in registry.list_models()

    def test_delete_version(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        v1 = registry.save_model(_DummyModel(), "multi-ver")
        v2 = registry.save_model(_DummyModel(), "multi-ver")

        registry.delete_version("multi-ver", v1)
        versions = registry.list_versions("multi-ver")
        assert v1 not in versions
        assert v2 in versions

    def test_save_and_load_experiment(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        exp = Experiment(name="my-exp")
        exp.log_run(model_name="m1", config={"a": 1}, metrics={"mae": 1.0})
        exp.log_run(model_name="m2", config={"a": 2}, metrics={"mae": 0.5})

        registry.save_experiment(exp)
        loaded = registry.load_experiment("my-exp")

        assert loaded.name == "my-exp"
        assert len(loaded.runs) == 2
        assert loaded.runs[0].metrics["mae"] == 1.0

    def test_list_experiments(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)

        exp1 = Experiment(name="exp-a")
        exp1.log_run(model_name="m", config={}, metrics={"mae": 1.0})
        registry.save_experiment(exp1)

        exp2 = Experiment(name="exp-b")
        exp2.log_run(model_name="m", config={}, metrics={"mae": 2.0})
        registry.save_experiment(exp2)

        names = registry.list_experiments()
        assert set(names) == {"exp-a", "exp-b"}

    def test_log_run_to_registry(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        model = _DummyModel(bias=1.0)
        model.fit(_make_ts())

        run = registry.log_run(
            experiment_name="auto-exp",
            model=model,
            model_name="dummy",
            config={"bias": 1.0},
            metrics={"mae": 0.3},
            save_model=True,
        )

        assert run.run_id is not None
        # Experiment was created
        exp = registry.load_experiment("auto-exp")
        assert len(exp.runs) == 1
        # Model was saved
        loaded = registry.load_model("dummy", version=run.run_id)
        assert loaded.bias == 1.0

    def test_reproducibility_seed_tracking(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)

        registry.log_run(
            experiment_name="seed-exp",
            model=_DummyModel(),
            model_name="dummy",
            config={"bias": 0.0, "seed": 42},
            metrics={"mae": 0.1},
        )

        exp = registry.load_experiment("seed-exp")
        assert exp.runs[0].config["seed"] == 42
