"""Experiment and run tracking for time series models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl


@dataclass
class Run:
    """A single model training/evaluation run."""

    run_id: str
    model_name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Run:
        return cls(
            run_id=d["run_id"],
            model_name=d["model_name"],
            config=d["config"],
            metrics=d["metrics"],
            timestamp=d.get("timestamp", ""),
            tags=d.get("tags", {}),
        )


@dataclass
class Experiment:
    """A collection of runs for comparison."""

    name: str
    runs: list[Run] = field(default_factory=list)

    def log_run(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> Run:
        run = Run(
            run_id=run_id or uuid.uuid4().hex[:12],
            model_name=model_name,
            config=config,
            metrics=metrics,
            tags=tags or {},
        )
        self.runs.append(run)
        return run

    def best_run(self, metric: str, *, higher_is_better: bool = False) -> Run:
        if not self.runs:
            raise ValueError("no runs logged — cannot determine best run")
        key = (lambda r: r.metrics[metric]) if not higher_is_better else (lambda r: -r.metrics[metric])
        return min(self.runs, key=key)

    def leaderboard(self, metric: str, *, higher_is_better: bool = False) -> pl.DataFrame:
        rows = []
        for r in self.runs:
            row: dict[str, Any] = {"run_id": r.run_id, "model_name": r.model_name}
            row.update(r.metrics)
            rows.append(row)
        df = pl.DataFrame(rows)
        return df.sort(metric, descending=higher_is_better)

    def to_dataframe(self) -> pl.DataFrame:
        rows = []
        for r in self.runs:
            row: dict[str, Any] = {
                "run_id": r.run_id,
                "model_name": r.model_name,
                "timestamp": r.timestamp,
            }
            row.update(r.metrics)
            rows.append(row)
        return pl.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Experiment:
        exp = cls(name=d["name"])
        exp.runs = [Run.from_dict(r) for r in d.get("runs", [])]
        return exp
