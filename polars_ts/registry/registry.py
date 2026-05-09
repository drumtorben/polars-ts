"""Model registry for saving, loading, and versioning time series models."""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polars_ts.registry.experiment import Experiment, Run


class ModelRegistry:
    """File-system-backed model registry with versioning and experiment tracking.

    Parameters
    ----------
    root
        Root directory for the registry. Models are stored under
        ``root/models/<name>/<version>/`` and experiments under
        ``root/experiments/<name>.json``.

    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._models_dir = self._root / "models"
        self._experiments_dir = self._root / "experiments"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._experiments_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        name: str,
        *,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a model to disk.

        Returns the version string used.
        """
        version = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        model_dir = self._models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        meta = metadata or {}
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return version

    def load_model(self, name: str, *, version: str | None = None) -> Any:
        """Load a model from disk. Loads latest version if *version* is None."""
        name_dir = self._models_dir / name
        if not name_dir.exists():
            raise FileNotFoundError(f"model {name!r} not found in registry")

        if version is None:
            # Pick the latest by mtime, then name as tiebreaker (for equal timestamps)
            versions = sorted(name_dir.iterdir(), key=lambda p: (p.stat().st_mtime, p.name))
            if not versions:
                raise FileNotFoundError(f"no versions found for model {name!r}")
            version_dir = versions[-1]
        else:
            version_dir = name_dir / version
            if not version_dir.exists():
                raise FileNotFoundError(f"version {version!r} not found for model {name!r}")

        with open(version_dir / "model.pkl", "rb") as f:
            return pickle.load(f)  # noqa: S301

    def get_metadata(self, name: str, *, version: str | None = None) -> dict[str, Any]:
        """Retrieve metadata for a saved model."""
        name_dir = self._models_dir / name
        if not name_dir.exists():
            raise FileNotFoundError(f"model {name!r} not found in registry")

        if version is None:
            versions = sorted(name_dir.iterdir(), key=lambda p: (p.stat().st_mtime, p.name))
            if not versions:
                raise FileNotFoundError(f"no versions found for model {name!r}")
            version_dir = versions[-1]
        else:
            version_dir = name_dir / version

        meta_path = version_dir / "metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def list_models(self) -> list[str]:
        """Return names of all registered models."""
        if not self._models_dir.exists():
            return []
        return sorted(p.name for p in self._models_dir.iterdir() if p.is_dir())

    def list_versions(self, name: str) -> list[str]:
        """Return all version strings for a given model."""
        name_dir = self._models_dir / name
        if not name_dir.exists():
            return []
        return sorted(p.name for p in name_dir.iterdir() if p.is_dir())

    def delete_model(self, name: str) -> None:
        """Remove a model and all its versions."""
        name_dir = self._models_dir / name
        if name_dir.exists():
            shutil.rmtree(name_dir)

    def delete_version(self, name: str, version: str) -> None:
        """Remove a specific version of a model."""
        version_dir = self._models_dir / name / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

    # ------------------------------------------------------------------
    # Experiment persistence
    # ------------------------------------------------------------------

    def save_experiment(self, experiment: Experiment) -> None:
        """Persist an experiment (all runs) to disk."""
        path = self._experiments_dir / f"{experiment.name}.json"
        with open(path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def load_experiment(self, name: str) -> Experiment:
        """Load a previously saved experiment."""
        path = self._experiments_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"experiment {name!r} not found")
        with open(path) as f:
            return Experiment.from_dict(json.load(f))

    def list_experiments(self) -> list[str]:
        """Return names of all saved experiments."""
        if not self._experiments_dir.exists():
            return []
        return sorted(p.stem for p in self._experiments_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Convenience: log run + optionally save model in one call
    # ------------------------------------------------------------------

    def log_run(
        self,
        *,
        experiment_name: str,
        model: Any,
        model_name: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        save_model: bool = False,
    ) -> Run:
        """Log a run to an experiment, optionally saving the model.

        If the experiment does not exist yet, it is created automatically.
        """
        # Load or create experiment
        try:
            exp = self.load_experiment(experiment_name)
        except FileNotFoundError:
            exp = Experiment(name=experiment_name)

        run = exp.log_run(
            model_name=model_name,
            config=config,
            metrics=metrics,
            tags=tags,
        )

        if save_model:
            self.save_model(model, model_name, version=run.run_id)

        self.save_experiment(exp)
        return run
