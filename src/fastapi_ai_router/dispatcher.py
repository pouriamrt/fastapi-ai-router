"""Dispatcher: split LLM-returned args by location, URL-encode path args, and
issue an in-process HTTP loopback to the FastAPI app via httpx ASGITransport.

This is the same pattern FastAPI's TestClient uses, so existing auth dependencies,
middleware, validation, and exception handlers all run during dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI

from fastapi_ai_router.errors import MissingPathParams
from fastapi_ai_router.schema import RouteSpec


def split_by_location(
    args: Mapping[str, Any],
    locations: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Partition args by the recorded param_locations (path/query/body)."""
    path: dict[str, Any] = {}
    query: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for k, v in args.items():
        loc = locations.get(k)
        if loc == "path":
            path[k] = v
        elif loc == "query":
            query[k] = v
        elif loc == "body":
            body[k] = v
        # unknown locations silently dropped — defensive against LLM hallucinations.
    return path, query, body


def _forward(request_headers: Mapping[str, str], allowed: frozenset[str]) -> dict[str, str]:
    return {k: v for k, v in request_headers.items() if k.lower() in allowed}


def _unwrap_body_field_names(body: dict[str, Any], locations: Mapping[str, str]) -> dict[str, Any]:
    """Reverse the *_body collision rename when constructing the loopback body.

    A field is treated as a renamed-due-to-collision body field if all of:
      - the key ends with "_body"
      - locations[key] == "body"
      - the original (without _body suffix) is also in locations under a
        non-body location (path or query)
    """
    out: dict[str, Any] = {}
    for k, v in body.items():
        if k.endswith("_body") and locations.get(k) == "body":
            original = k[: -len("_body")]
            if original in locations and locations[original] != "body":
                out[original] = v
                continue
        out[k] = v
    return out


async def dispatch(
    *,
    spec: RouteSpec,
    args: dict[str, Any],
    app: FastAPI,
    request_headers: Mapping[str, str],
    forward: frozenset[str],
) -> httpx.Response:
    """Dispatch the LLM-chosen tool call as an in-process HTTP request."""
    path_args, query_args, body_args = split_by_location(args, spec.param_locations)
    missing = tuple(
        name
        for name, loc in spec.param_locations.items()
        if loc == "path" and name not in path_args
    )
    if missing:
        raise MissingPathParams(missing)

    encoded_path_args = {k: quote(str(v), safe="") for k, v in path_args.items()}
    url = spec.path_template.format(**encoded_path_args)

    body_args = _unwrap_body_field_names(body_args, spec.param_locations)
    headers = _forward(request_headers, forward)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://airouter.local",
    ) as client:
        return await client.request(
            method=spec.method,
            url=url,
            params=query_args or None,
            json=body_args if body_args else None,
            headers=headers,
        )


__all__ = ["dispatch", "split_by_location"]
