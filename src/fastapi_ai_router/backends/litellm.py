"""LiteLLMBackend — convenience adapter around litellm.acompletion.

Imported lazily so the core has no hard dependency on litellm. Install with
`pip install fastapi-ai-router[litellm]` to enable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    from litellm import acompletion  # type: ignore[import-not-found,unused-ignore]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LiteLLMBackend requires the 'litellm' extra. "
        "Install with: pip install fastapi-ai-router[litellm]"
    ) from exc

from fastapi_ai_router.backends import LLMBackend, Message, ToolCall, ToolDef
from fastapi_ai_router.errors import LLMBackendError


@dataclass
class LiteLLMBackend(LLMBackend):
    """Default convenience backend. Forwards (messages, tools) to LiteLLM
    in OpenAI-compatible function-calling format and parses the response."""

    model: str
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None:
        try:
            response = await acompletion(
                model=self.model,
                messages=list(messages),
                tools=list(tools) or None,
                **self.extra_kwargs,
            )
        except BaseException as exc:
            raise LLMBackendError(f"LiteLLM call failed: {exc}", upstream=exc) from exc

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Any) -> ToolCall | None:
        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            return None
        first = tool_calls[0]
        func = first["function"]
        name = func["name"]
        try:
            args = json.loads(func.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise LLMBackendError(
                f"LiteLLM returned non-JSON arguments: {func.get('arguments')!r}",
                upstream=exc,
            ) from exc
        usage = response.get("usage") or {}
        return ToolCall(
            name=name,
            args=args,
            reasoning=choice.get("content"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=response.get("model") or "",
        )


__all__ = ["LiteLLMBackend"]
