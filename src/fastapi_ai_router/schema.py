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

from fastapi.routing import APIRoute
from pydantic import BaseModel, TypeAdapter

from fastapi_ai_router.decorator import AI_ROUTE_ATTR, AIRouteMeta

# "whole_body": a lone, non-embedded body param. FastAPI reads the request body
# as that param's bare value, not as {name: value}.
ParamLocation = Literal["path", "query", "body", "whole_body"]


@dataclass(frozen=True)
class RouteSpec:
    name: str
    description: str
    method: str
    path_template: str
    parameters_schema: dict[str, Any]
    param_locations: dict[str, ParamLocation]
    handler: Callable[..., Any]


def _is_form_or_multipart_route(route: APIRoute) -> bool:
    """Detect routes whose body content-type isn't JSON.

    FastAPI marks Form/File params on dependant.body_params; a body params list
    with `media_type` not application/json signals form/multipart.
    """
    body_params = getattr(route.dependant, "body_params", []) or []
    for param in body_params:
        media_type = getattr(getattr(param, "field_info", None), "media_type", None)
        if media_type and media_type != "application/json":
            return True
    return False


def _description_from_route(route: APIRoute) -> str:
    """Resolve LLM-facing description: @ai_route → docstring → empty."""
    fn = route.endpoint
    meta: AIRouteMeta | None = getattr(fn, AI_ROUTE_ATTR, None)
    if meta and meta.description:
        return meta.description
    return (fn.__doc__ or "").strip()


def _annotation_of(param: Any) -> Any:
    """Best-effort extraction of the type annotation from a FastAPI ModelField.

    FastAPI's internal ModelField (in fastapi._compat for Pydantic v2) exposes:
      - .field_info: FieldInfo (with .annotation)
      - .type_: the resolved annotation
    Older versions may differ. We try both, fall back to None.
    """
    fi = getattr(param, "field_info", None)
    if fi is not None:
        ann = getattr(fi, "annotation", None)
        if ann is not None:
            return ann
    return getattr(param, "type_", None)


def _is_required(param: Any) -> bool:
    """Best-effort extraction of required-ness from a FastAPI ModelField."""
    if hasattr(param, "required"):
        try:
            return bool(param.required)
        except Exception:
            pass
    fi = getattr(param, "field_info", None)
    if fi is not None and hasattr(fi, "is_required"):
        try:
            return bool(fi.is_required())
        except Exception:
            pass
    return False


def _is_embedded(param: Any) -> bool:
    """True for `Body(embed=True)`: FastAPI then expects {name: value} even when alone."""
    return bool(getattr(getattr(param, "field_info", None), "embed", False))


def _field_schema(param: Any) -> dict[str, Any]:
    """Convert a FastAPI ModelField to a JSON Schema fragment via TypeAdapter."""
    annotation = _annotation_of(param)
    if annotation is None:
        return {}
    try:
        return TypeAdapter(annotation).json_schema()
    except Exception:
        return {}


def _build_parameters_schema(
    route: APIRoute,
) -> tuple[dict[str, Any], dict[str, ParamLocation]]:
    """Build the flat parameters_schema and param_locations for a route."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    locations: dict[str, ParamLocation] = {}
    defs: dict[str, Any] = {}

    # Path parameters: always required.
    for param in route.dependant.path_params or []:
        properties[param.name] = _field_schema(param)
        required.append(param.name)
        locations[param.name] = "path"

    # Query parameters: required only if FastAPI says so.
    for param in route.dependant.query_params or []:
        properties[param.name] = _field_schema(param)
        if _is_required(param):
            required.append(param.name)
        locations[param.name] = "query"

    # Body params mirror FastAPI's wire shape: a lone, non-embedded model is
    # flattened to its fields; any other lone, non-embedded param is the whole
    # body; embedded or multiple params are keyed by name.
    body_params = list(route.dependant.body_params or [])
    if len(body_params) == 1:
        bp = body_params[0]
        annotation = _annotation_of(bp)
        embedded = _is_embedded(bp)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and not embedded:
            schema = annotation.model_json_schema()
            # Field schemas point at "#/$defs/..."; hoist the defs so those refs resolve.
            defs.update(schema.get("$defs") or {})
            for fname, fschema in (schema.get("properties") or {}).items():
                emit_name = fname
                if emit_name in locations:  # collision with path/query
                    emit_name = f"{fname}_body"
                properties[emit_name] = fschema
                if fname in (schema.get("required") or []):
                    required.append(emit_name)
                locations[emit_name] = "body"
        else:
            # List / dict / scalar / optional model: exposed to the LLM under the param name.
            wrap_name = bp.name
            if wrap_name in locations:
                wrap_name = f"{bp.name}_body"
            properties[wrap_name] = _field_schema(bp)
            if _is_required(bp):
                required.append(wrap_name)
            locations[wrap_name] = "body" if embedded else "whole_body"
    else:
        # Multiple Body() params — each becomes a top-level body field.
        for bp in body_params:
            emit_name = bp.name
            if emit_name in locations:
                emit_name = f"{bp.name}_body"
            properties[emit_name] = _field_schema(bp)
            if _is_required(bp):
                required.append(emit_name)
            locations[emit_name] = "body"

    schema_obj: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if defs:
        schema_obj["$defs"] = defs
    return schema_obj, locations


def route_to_spec(route: APIRoute) -> RouteSpec | None:
    """Project a FastAPI APIRoute into a RouteSpec.

    Returns None if the route is excluded from AI exposure (form/multipart
    bodies, file uploads, or routes with no operations).
    """
    if not isinstance(route, APIRoute):
        return None
    if _is_form_or_multipart_route(route):
        return None
    if not route.methods:
        return None

    method = next(iter(route.methods)).upper()
    parameters_schema, param_locations = _build_parameters_schema(route)
    description = _description_from_route(route)
    name = route.name or route.unique_id or f"{method.lower()}_{route.path}"

    return RouteSpec(
        name=name,
        description=description,
        method=method,
        path_template=route.path,
        parameters_schema=parameters_schema,
        param_locations=param_locations,
        handler=route.endpoint,
    )


__all__ = ["ParamLocation", "RouteSpec", "route_to_spec"]
