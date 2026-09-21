"""Exception hierarchy for fastapi-ai-router.

All library errors derive from AIRouterError so callers can catch them with
a single except clause.
"""

from __future__ import annotations


class AIRouterError(Exception):
    """Base — all library errors derive from this."""


class NoRouteMatched(AIRouterError):
    """LLM declined to call any tool. Carries the LLM's text reply (if any)."""

    def __init__(self, llm_text: str | None = None) -> None:
        self.llm_text = llm_text
        super().__init__(llm_text or "LLM did not select a route.")


class UnknownTool(AIRouterError):
    """LLM called a tool name we did not expose."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"LLM called unknown tool: {tool_name!r}")


class LLMBackendError(AIRouterError):
    """LLM call itself failed (timeout, rate limit, invalid response).

    Wraps the upstream exception so it can be inspected.
    """

    def __init__(self, detail: str, upstream: BaseException | None = None) -> None:
        self.upstream = upstream
        super().__init__(detail)


class DispatchError(AIRouterError):
    """Loopback HTTP call failed for non-route-logic reasons (transport error)."""


class ToolSchemaTooLarge(AIRouterError):
    """Combined tool definitions exceed the configured token budget."""

    def __init__(self, tool_count: int, approx_tokens: int) -> None:
        self.tool_count = tool_count
        self.approx_tokens = approx_tokens
        super().__init__(f"Tool schema is too large: {tool_count} tools, ~{approx_tokens} tokens.")


class MissingPathParams(AIRouterError):
    """Backend left out one or more path parameters, so the URL can't be built."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(f"Missing path parameters: {', '.join(missing)}")
