"""LLM backend protocol and shared context for the agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class LLMBackend(Protocol):
    """Abstraction over any LLM provider.

    Implement ``complete`` to plug in OpenAI, Anthropic, local models, etc.
    """

    def complete(self, prompt: str) -> str: ...


class RuleBasedBackend:
    """Default backend that uses deterministic heuristics instead of an LLM."""

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return ""


@dataclass
class AgentContext:
    """Shared mutable state passed between agents in a pipeline run."""

    data: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, message: str) -> None:
        self.history.append({"agent": agent, "message": message})
