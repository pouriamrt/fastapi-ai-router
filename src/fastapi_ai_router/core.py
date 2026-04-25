"""AIRouter — wires introspection, LLM call, dispatch, and envelope into a single FastAPI route."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request, Response

from fastapi_ai_router.backends import LLMBackend, Message, ToolCall
from fastapi_ai_router.dispatcher import dispatch
from fastapi_ai_router.envelope import is_raw_requested, wrap_envelope
from fastapi_ai_router.errors import (
    AIRouterError,
    DispatchError,
)
from fastapi_ai_router.introspection import Mode, ModeConfig, build_registry, build_tools
from fastapi_ai_router.observability import (
    Decision,
    DecisionHook,
    ErrorEvent,
    ErrorHook,
)
from fastapi_ai_router.schema import RouteSpec

DEFAULT_FORWARD_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "x-api-key",
        "x-forwarded-for",
        "x-request-id",
    }
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a router. Choose exactly one tool that best satisfies the user's "
    "request, and fill in its arguments using only what the user provided. If "
    "no tool fits, do not call any tool."
)

logger = logging.getLogger("fastapi_ai_router")


class AIRouter:
    """Add a natural-language router endpoint to a FastAPI app.

    Walks ``app.routes`` lazily on first request, projects them to LLM tool
    schemas, asks ``llm`` to pick one, and dispatches the call internally.
    """

    def __init__(
        self,
        app: FastAPI,
        *,
        llm: LLMBackend,
        mode: Mode = "decorator",
        tag: str = "ai",
        exclude: Sequence[str] | None = None,
        endpoint: str = "/ai",
        dependencies: Sequence[Any] | None = None,
        raw_query_param: str = "raw",
        forward_headers: frozenset[str] = DEFAULT_FORWARD_HEADERS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_decision: DecisionHook | None = None,
        on_error: ErrorHook | None = None,
        debug: bool = False,
    ) -> None:
        self._app = app
        self._llm = llm
        self._mode_cfg = ModeConfig(
            mode=mode,
            tag=tag,
            exclude=tuple(exclude or ()),
        )
        self._endpoint = endpoint
        self._raw_param = raw_query_param
        self._forward_headers = forward_headers
        self._system_prompt = system_prompt
        self._on_decision = on_decision
        self._on_error = on_error
        self._debug = debug

        self._registry: dict[str, RouteSpec] | None = None

        app.add_api_route(
            path=endpoint,
            endpoint=self._handle,
            methods=["POST"],
            dependencies=list(dependencies or []),
            include_in_schema=True,
            name="ai_router",
        )

    def rebuild(self) -> None:
        """Discard the cached registry; next request rebuilds from app.routes."""
        self._registry = None

    def _ensure_registry(self) -> dict[str, RouteSpec]:
        if self._registry is None:
            self._registry = build_registry(self._app, self._mode_cfg)
        return self._registry

    async def _handle(self, request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json_response(422, {"error": "invalid_json"})
        if not isinstance(body, dict):
            return _json_response(422, {"error": "invalid_body"})
        query = body.get("query", "")
        if not isinstance(query, str) or not query:
            return _json_response(422, {"error": "missing_query"})

        registry = self._ensure_registry()
        tools = build_tools(registry)

        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        # ---- LLM call ----
        llm_started = time.perf_counter()
        try:
            tool_call = await self._llm.call(messages=messages, tools=tools)
        except AIRouterError:
            raise
        except BaseException as exc:
            await self._fire_error(request_id, query, "llm_backend_error", str(exc), exc)
            return _json_response(
                502,
                {
                    "error": "llm_backend_error",
                    "detail": str(exc),
                    "retryable": True,
                },
            )
        llm_latency_ms = int((time.perf_counter() - llm_started) * 1000)

        # ---- branching ----
        if tool_call is None:
            await self._fire_error(request_id, query, "no_route_matched", "", None)
            return _json_response(
                422,
                {
                    "error": "no_route_matched",
                    "reasoning": None,
                    "available_tools": [t["function"]["name"] for t in tools],
                },
            )

        spec = registry.get(tool_call.name)
        if spec is None:
            await self._fire_error(
                request_id, query, "unknown_tool", tool_call.name, None
            )
            return _json_response(
                422,
                {
                    "error": "unknown_tool",
                    "tool_name": tool_call.name,
                    "available_tools": [t["function"]["name"] for t in tools],
                },
            )

        # ---- dispatch ----
        dispatch_started = time.perf_counter()
        try:
            response = await dispatch(
                spec=spec,
                args=tool_call.args,
                app=self._app,
                request_headers=dict(request.headers),
                forward=self._forward_headers,
            )
        except BaseException as exc:
            await self._fire_error(request_id, query, "dispatch_error", str(exc), exc)
            raise DispatchError(str(exc)) from exc
        dispatch_latency_ms = int((time.perf_counter() - dispatch_started) * 1000)

        # ---- response shape ----
        endpoint_label = f"{spec.method} {spec.path_template}"
        if is_raw_requested(dict(request.query_params), raw_param=self._raw_param):
            ret = Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    "content-type": response.headers.get(
                        "content-type", "application/json"
                    )
                },
            )
        else:
            envelope = wrap_envelope(
                endpoint=endpoint_label,
                args=tool_call.args,
                reasoning=tool_call.reasoning,
                response=response,
            )
            ret = _json_response(response.status_code, envelope)

        await self._fire_decision(
            request_id=request_id,
            query=query,
            tool_call=tool_call,
            llm_latency_ms=llm_latency_ms,
            dispatch_latency_ms=dispatch_latency_ms,
            result_status=response.status_code,
        )
        return ret

    async def _fire_decision(
        self,
        *,
        request_id: str,
        query: str,
        tool_call: ToolCall,
        llm_latency_ms: int,
        dispatch_latency_ms: int | None,
        result_status: int | None,
    ) -> None:
        if self._on_decision is None:
            return
        decision = Decision(
            request_id=request_id,
            query=query,
            tool_name=tool_call.name,
            args=tool_call.args,
            reasoning=tool_call.reasoning,
            model=tool_call.model,
            prompt_tokens=tool_call.prompt_tokens,
            completion_tokens=tool_call.completion_tokens,
            llm_latency_ms=llm_latency_ms,
            dispatch_latency_ms=dispatch_latency_ms,
            result_status=result_status,
        )
        try:
            await self._on_decision(decision)
        except Exception:
            logger.exception("on_decision hook raised; swallowing")

    async def _fire_error(
        self,
        request_id: str,
        query: str,
        error_type: str,
        error_detail: str,
        upstream: BaseException | None,
    ) -> None:
        if self._on_error is None:
            return
        event = ErrorEvent(
            request_id=request_id,
            query=query,
            error_type=error_type,
            error_detail=error_detail,
            upstream=upstream,
        )
        try:
            await self._on_error(event)
        except Exception:
            logger.exception("on_error hook raised; swallowing")


def _json_response(status: int, body: dict[str, Any]) -> Response:
    return Response(
        content=json.dumps(body).encode(),
        status_code=status,
        media_type="application/json",
    )


__all__ = ["DEFAULT_FORWARD_HEADERS", "DEFAULT_SYSTEM_PROMPT", "AIRouter"]
