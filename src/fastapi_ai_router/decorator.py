"""The @ai_route decorator — attaches AI-routing metadata to a FastAPI route handler.

The decorator is a pure annotation: it does NOT modify call behavior. Introspection
reads the attached AIRouteMeta to decide whether (and how) to expose the route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

AI_ROUTE_ATTR = "__ai_route__"

F = TypeVar("F", bound=Callable[..., object])


@dataclass(frozen=True)
class AIRouteMeta:
    """Metadata attached by @ai_route to a route handler function."""

    description: str | None = None
    expose: bool = True


def ai_route(
    *, description: str | None = None, expose: bool = True
) -> Callable[[F], F]:
    """Mark a FastAPI route as AI-callable.

    Arguments:
        description: human-readable description used as the LLM-facing tool
            description. Falls back to the function's docstring if omitted.
        expose: kill switch. If False, the route is excluded from AI exposure
            in *every* mode (decorator, tag, all). Useful for marking sensitive
            routes as never-AI-callable without relying on path patterns.
    """
    meta = AIRouteMeta(description=description, expose=expose)

    def decorator(fn: F) -> F:
        setattr(fn, AI_ROUTE_ATTR, meta)
        return fn

    return decorator
