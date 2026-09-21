"""Introspection: walk a FastAPI app's routes, apply the mode filter, and produce
both a registry (name → RouteSpec) and an OpenAI-compatible list of tool definitions.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI
from fastapi.routing import APIRoute

from fastapi_ai_router.backends import FunctionDef, ToolDef
from fastapi_ai_router.decorator import AI_ROUTE_ATTR, AIRouteMeta
from fastapi_ai_router.schema import RouteSpec, route_to_spec

Mode = Literal["decorator", "tag", "all"]


@dataclass(frozen=True)
class ModeConfig:
    mode: Mode = "decorator"
    tag: str = "ai"
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("decorator", "tag", "all"):
            raise ValueError(f"mode must be one of 'decorator', 'tag', 'all'; got {self.mode!r}")


def _ai_meta(route: APIRoute) -> AIRouteMeta | None:
    return getattr(route.endpoint, AI_ROUTE_ATTR, None)


def _is_kill_switched(route: APIRoute) -> bool:
    meta = _ai_meta(route)
    return meta is not None and meta.expose is False


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def _passes_mode_filter(route: APIRoute, cfg: ModeConfig) -> bool:
    if cfg.mode == "decorator":
        meta = _ai_meta(route)
        return meta is not None and meta.expose
    if cfg.mode == "tag":
        return cfg.tag in (route.tags or [])
    # mode == "all"
    return not _matches_any(route.path, cfg.exclude)


def _ai_eligible(route: APIRoute) -> bool:
    """True if the route is even a candidate (regardless of mode)."""
    if not isinstance(route, APIRoute):
        return False
    if route.include_in_schema is False:
        return False
    return not _is_kill_switched(route)


def build_registry(app: FastAPI, cfg: ModeConfig) -> dict[str, RouteSpec]:
    """Walk app.routes and produce {tool_name: RouteSpec} for AI-exposed routes."""
    registry: dict[str, RouteSpec] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not _ai_eligible(route):
            continue
        if not _passes_mode_filter(route, cfg):
            continue
        spec = route_to_spec(route)
        if spec is None:  # form/multipart/excluded
            continue

        name = spec.name
        if name in registry:
            base = f"{name}_{spec.method.lower()}"
            suffix = hashlib.sha1(spec.path_template.encode()).hexdigest()[:4]
            name = f"{base}_{suffix}"
            spec = RouteSpec(
                name=name,
                description=spec.description,
                method=spec.method,
                path_template=spec.path_template,
                parameters_schema=spec.parameters_schema,
                param_locations=spec.param_locations,
                handler=spec.handler,
            )
        registry[name] = spec
    return registry


def build_tools(registry: dict[str, RouteSpec]) -> list[ToolDef]:
    """Convert a registry to the OpenAI-compatible tool list."""
    return [
        ToolDef(
            type="function",
            function=FunctionDef(
                name=spec.name,
                description=spec.description or "",
                parameters=spec.parameters_schema,
            ),
        )
        for spec in registry.values()
    ]


__all__ = ["Mode", "ModeConfig", "build_registry", "build_tools"]
