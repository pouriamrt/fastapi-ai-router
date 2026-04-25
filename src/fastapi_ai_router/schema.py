"""Route specification — what AIRouter knows about a single FastAPI route.

A RouteSpec is an immutable projection of a FastAPI route + OpenAPI operation.
The flat parameters_schema is what the LLM sees (a single JSON Schema). The
param_locations map records where each top-level field came from so the
dispatcher can un-flatten when calling the route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ParamLocation = Literal["path", "query", "body"]


@dataclass(frozen=True)
class RouteSpec:
    name: str
    description: str
    method: str
    path_template: str
    parameters_schema: dict[str, Any]
    param_locations: dict[str, ParamLocation]
    handler: Callable[..., Any]


__all__ = ["ParamLocation", "RouteSpec"]
