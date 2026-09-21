"""JevBackend: route requests with TypeSafe's Jev classifier instead of a generative LLM.

Jev picks from options; it never writes text. The route is a Choice over tool
names, and each argument is a Choice over candidate values cut from the query
(numbers, word spans, enum values). Every question goes out in one request, and
code reads the answers for the route Jev picked. Install with
`pip install fastapi-ai-router[jev]`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

try:
    from typesafe_sdk import Choice
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "JevBackend requires the 'jev' extra. Install with: pip install fastapi-ai-router[jev]"
    ) from exc

from fastapi_ai_router.backends import ToolDef

NOT_STATED = "not_stated"
NO_ROUTE = "none_of_the_above"
ROUTE_QID = "route"
MAX_CHOICE_OPTIONS = 255  # the API rejects more with 400 "Too many choices"
ROUTE_INSTRUCTIONS = "Which API route should handle this user request?"

ParamKind = Literal["integer", "number", "boolean", "string", "enum"]
_SCALAR_KINDS = frozenset({"integer", "number", "boolean", "string"})

_WORD = re.compile(r"[\w'-]+")
_INT = re.compile(r"\d+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ArgSlot:
    """One (route, param) pair and the candidate values offered to Jev."""

    qid: str
    route: str
    param: str
    kind: ParamKind | None  # None: a type Jev can't extract (object, array, ...)
    required: bool
    options: tuple[str, ...]
    enum_values: tuple[Any, ...] = ()


def _resolve(schema: Mapping[str, Any], root_defs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Unwrap Optional (anyOf with null) and a local $ref. None if unresolvable."""
    defs = {**root_defs, **(schema.get("$defs") or {})}
    variants = schema.get("anyOf")
    if isinstance(variants, list):
        non_null = [v for v in variants if v.get("type") != "null"]
        if len(non_null) != 1:
            return None
        schema = non_null[0]
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = defs.get(ref.removeprefix("#/$defs/"))
        return target if isinstance(target, Mapping) else None
    return schema


def classify(
    schema: Mapping[str, Any], root_defs: Mapping[str, Any]
) -> tuple[ParamKind | None, tuple[Any, ...]]:
    """What Jev can extract for a param's JSON schema, plus its enum values."""
    resolved = _resolve(schema, root_defs)
    if resolved is None:
        return None, ()
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        return "enum", tuple(enum)
    kind = resolved.get("type")
    if kind in _SCALAR_KINDS:
        return kind, ()
    return None, ()


def candidates(
    query: str, kind: ParamKind, enum_values: tuple[Any, ...], max_span_words: int
) -> tuple[str, ...]:
    """Candidate values for one param, capped to fit a single Choice."""
    if kind == "enum":
        spans = [str(v) for v in enum_values]
    elif kind == "boolean":
        spans = ["true", "false"]
    elif kind == "integer":
        spans = _INT.findall(query)
    elif kind == "number":
        spans = _NUMBER.findall(query)
    else:
        words = _WORD.findall(query)
        spans = [
            " ".join(words[i:j])
            for i in range(len(words))
            for j in range(i + 1, min(len(words), i + max_span_words) + 1)
        ]
    unique = [s for s in dict.fromkeys(spans) if s != NOT_STATED]
    # ponytail: long queries lose their late spans at the cap; extract per route
    # in a second call if that bites.
    return tuple(unique[: MAX_CHOICE_OPTIONS - 1])


def build_slots(query: str, tools: list[ToolDef], max_span_words: int) -> list[ArgSlot]:
    """One slot per (route, param) across every tool."""
    slots: list[ArgSlot] = []
    for tool in tools:
        params = tool["function"]["parameters"]
        root_defs = params.get("$defs") or {}
        required = set(params.get("required") or [])
        for name, schema in (params.get("properties") or {}).items():
            kind, enum_values = classify(schema, root_defs)
            options = candidates(query, kind, enum_values, max_span_words) if kind else ()
            slots.append(
                ArgSlot(
                    qid=f"arg{len(slots)}",
                    route=tool["function"]["name"],
                    param=name,
                    kind=kind,
                    required=name in required,
                    options=options,
                    enum_values=enum_values,
                )
            )
    return slots


def _arg_instructions(slot: ArgSlot) -> str:
    prefix = f"If this request is handled by route `{slot.route}`,"
    if slot.kind in ("enum", "boolean"):
        return f"{prefix} which option matches the value it gives for parameter `{slot.param}`?"
    return (
        f"{prefix} which exact words of the request are the value of "
        f"parameter `{slot.param}` ({slot.kind})?"
    )


def build_questions(tools: list[ToolDef], slots: list[ArgSlot]) -> dict[str, Choice]:
    """The route question plus one question per slot that has candidates."""
    # ponytail: asks every route's params up front (one round trip); split into
    # route-then-args calls if large apps hit Jev's 64k-token context limit.
    routes: dict[str, str | None] = {
        t["function"]["name"]: t["function"]["description"] or None for t in tools
    }
    routes[NO_ROUTE] = "The request fits none of the other routes."
    questions = {ROUTE_QID: Choice(instructions=ROUTE_INSTRUCTIONS, criteria=routes)}
    for slot in slots:
        if slot.options:
            criteria: dict[str, str | None] = dict.fromkeys(slot.options)
            criteria[NOT_STATED] = "The request does not state this value."
            questions[slot.qid] = Choice(instructions=_arg_instructions(slot), criteria=criteria)
    return questions


def coerce(slot: ArgSlot, raw: str) -> Any:
    """Turn Jev's chosen option into a typed value; None if it doesn't convert."""
    try:
        if slot.kind == "integer":
            return int(raw)
        if slot.kind == "number":
            return float(raw)
    except ValueError:
        return None
    if slot.kind == "boolean":
        return raw == "true"
    if slot.kind == "enum":
        return {str(v): v for v in slot.enum_values}.get(raw)
    return raw
