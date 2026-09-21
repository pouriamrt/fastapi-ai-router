"""LLM backend protocol and shared wire types.

The core depends only on this protocol — no vendor SDKs are imported here.
Backends translate the OpenAI-compatible wire shape (Message, ToolDef) to
whatever their underlying SDK expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class FunctionDef(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDef(TypedDict):
    type: Literal["function"]
    function: FunctionDef


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    model: str
    confidence: float | None = None


class LLMBackend(Protocol):
    """Protocol for LLM backends consumable by AIRouter.

    Implementations translate (messages, tools) to their vendor's
    function-calling API and return either a ToolCall (one selected tool)
    or None (model returned plain text without invoking a tool).
    """

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None: ...


__all__ = [
    "FunctionDef",
    "LLMBackend",
    "Message",
    "ToolCall",
    "ToolDef",
]
