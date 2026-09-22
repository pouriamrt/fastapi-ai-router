"""Observability primitives: Decision, ErrorEvent, async hook protocols.

The core fires hooks but takes no vendor dependencies. Users plug in
Langfuse, OpenTelemetry, Sentry, or anything else by passing async functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Decision:
    request_id: str
    query: str
    tool_name: str | None
    args: dict[str, Any]
    reasoning: str | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: int
    dispatch_latency_ms: int | None
    result_status: int | None
    confidence: float | None = None


@dataclass(frozen=True)
class ErrorEvent:
    request_id: str
    query: str
    error_type: str
    error_detail: str
    upstream: BaseException | None


# Positional-only, so a hook may name its parameter anything (`async def hook(d): ...`).
class DecisionHook(Protocol):
    async def __call__(self, decision: Decision, /) -> None: ...


class ErrorHook(Protocol):
    async def __call__(self, event: ErrorEvent, /) -> None: ...


__all__ = [
    "Decision",
    "DecisionHook",
    "ErrorEvent",
    "ErrorHook",
]
