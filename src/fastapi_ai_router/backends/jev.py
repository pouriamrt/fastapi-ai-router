"""JevBackend: route requests with TypeSafe's Jev classifier instead of a generative LLM.

Jev picks from options; it never writes text. The route is a Choice over tool
names, and each argument is a Choice over candidate values cut from the query
(numbers, word spans, enum values). Every question goes out in one request, and
code reads the answers for the route Jev picked. Install with
`pip install fastapi-ai-router[jev]`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        ChoiceAnswer,
        SystemOneResponse,
        TypeSafeError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "JevBackend requires the 'jev' extra. Install with: pip install fastapi-ai-router[jev]"
    ) from exc

from fastapi_ai_router.backends import LLMBackend, Message, ToolCall, ToolDef
from fastapi_ai_router.errors import LLMBackendError

NOT_STATED = "not_stated"
NO_ROUTE = "none_of_the_above"
ROUTE_QID = "route"
MAX_CHOICE_OPTIONS = 255  # the API rejects more with 400 "Too many choices"
ROUTE_INSTRUCTIONS = "Which API route should handle this user request?"

ParamKind = Literal["integer", "number", "boolean", "string", "enum"]
_SCALAR_KINDS = frozenset({"integer", "number", "boolean", "string"})

_WORD = re.compile(r"[\w'-]+")
_SIGN = r"(?:(?<![\w-])-)?"  # "-5", not the dash in "order-123" / "5-10"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)"  # "1,000" or "1000"
_INT = re.compile(rf"{_SIGN}{_DIGITS}")
_NUMBER = re.compile(rf"{_SIGN}(?:{_DIGITS}(?:\.\d+)?|\.\d+)")


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


def _string_spans(words: list[str], max_span_words: int) -> Iterator[str]:
    """Every contiguous word n-gram up to max_span_words long, in scan order."""
    for i in range(len(words)):
        upper = min(len(words), i + max_span_words) + 1
        for j in range(i + 1, upper):
            yield " ".join(words[i:j])


def candidates(
    query: str, kind: ParamKind, enum_values: tuple[Any, ...], max_span_words: int
) -> tuple[str, ...]:
    """Candidate values for one param, capped to fit a single Choice."""
    spans: Iterable[str]
    if kind == "enum":
        spans = (str(v) for v in enum_values)
    elif kind == "boolean":
        spans = ("true", "false")
    elif kind == "integer":
        spans = _INT.findall(query)
    elif kind == "number":
        spans = _NUMBER.findall(query)
    else:
        # ponytail: spans are generated lazily and stop at the cap below, so a
        # long query never materializes every n-gram; a max query length in
        # core is the real bound (follow-up, not this wave).
        spans = _string_spans(_WORD.findall(query), max_span_words)
    cap = MAX_CHOICE_OPTIONS - 1
    unique: dict[str, None] = {}
    for span in spans:
        if span == NOT_STATED or span in unique:
            continue
        unique[span] = None
        if len(unique) == cap:
            break
    return tuple(unique)


def build_slots(query: str, tools: list[ToolDef], max_span_words: int) -> list[ArgSlot]:
    """One slot per (route, param) across every tool."""
    slots: list[ArgSlot] = []
    # Every string slot (no enum) shares identical candidates; every enum slot
    # with the same values shares identical candidates too. Compute each once.
    cache: dict[tuple[ParamKind, tuple[Any, ...]], tuple[str, ...]] = {}
    for tool in tools:
        params = tool["function"]["parameters"]
        root_defs = params.get("$defs") or {}
        required = set(params.get("required") or [])
        for name, schema in (params.get("properties") or {}).items():
            kind, enum_values = classify(schema, root_defs)
            options: tuple[str, ...] = ()
            if kind is not None:
                key = (kind, enum_values)
                if key not in cache:
                    cache[key] = candidates(query, kind, enum_values, max_span_words)
                options = cache[key]
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
            return int(raw.replace(",", ""))
        if slot.kind == "number":
            return float(raw.replace(",", ""))
    except ValueError:
        return None
    if slot.kind == "boolean":
        return raw == "true"
    if slot.kind == "enum":
        return {str(v): v for v in slot.enum_values}.get(raw)
    return raw


def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def _answer(response: SystemOneResponse, qid: str) -> ChoiceAnswer:
    answer = response.choices.get(qid)
    if answer is None:
        raise LLMBackendError(f"Jev response is missing answer {qid!r}")
    return answer


def collect_args(response: SystemOneResponse, slots: list[ArgSlot]) -> tuple[dict[str, Any], bool]:
    """Typed args for one route, and whether a required param came back empty."""
    args: dict[str, Any] = {}
    missing_required = False
    for slot in slots:
        value = None
        if slot.options:
            choice = _answer(response, slot.qid).choice
            value = None if choice == NOT_STATED else coerce(slot, choice)
        if value is None:
            missing_required = missing_required or slot.required
        else:
            args[slot.param] = value
    return args, missing_required


@dataclass
class JevBackend(LLMBackend):
    """Routes with Jev in one request; optionally hands the hard cases to an LLM.

    `fallback` runs only when Jev is unsure of the route (it gets every tool) or
    the chosen route needs a value Jev couldn't supply (it gets that route's tool
    only). With no fallback, the router never calls an LLM.
    """

    model: str = "jev-latest"
    api_key: str | None = field(default=None, repr=False)  # None: the SDK reads TYPESAFE_API_KEY
    min_confidence: float = 0.5
    fallback: LLMBackend | None = None
    max_span_words: int = 6
    client: AsyncTypeSafeClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)
    _client_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None:
        query = _last_user_content(messages)
        slots = build_slots(query, tools, self.max_span_words)
        response = await self._ask(query, build_questions(tools, slots))
        route = _answer(response, ROUTE_QID)
        if route.confidence < self.min_confidence:
            return await self._fall_back(messages, tools, confidence=None)
        if route.choice == NO_ROUTE:
            return None
        route_slots = [s for s in slots if s.route == route.choice]
        args, missing_required = collect_args(response, route_slots)
        if missing_required and self.fallback is not None:
            narrowed = [t for t in tools if t["function"]["name"] == route.choice]
            return await self._fall_back(messages, narrowed, confidence=route.confidence)
        return ToolCall(
            name=route.choice,
            args=args,
            reasoning=None,
            # Usage.*_tokens is Optional in the SDK; Jev always reports them in practice.
            prompt_tokens=response.usage.input_tokens or 0,
            completion_tokens=response.usage.output_tokens or 0,
            model=response.model,
            confidence=route.confidence,
        )

    async def _ask(self, query: str, questions: dict[str, Choice]) -> SystemOneResponse:
        try:
            loop = asyncio.get_running_loop()
            if self.client is None or (self._owns_client and self._client_loop is not loop):
                # ponytail: one shared client per event loop, never closed (AIRouter
                # has no shutdown hook); an injected client is never replaced.
                self.client = AsyncTypeSafeClient(api_key=self.api_key, model=self.model)
                self._owns_client, self._client_loop = True, loop
            client = self.client
            return await client.system_one(query, questions)
        except TypeSafeError as exc:
            raise LLMBackendError(f"Jev call failed: {exc}", upstream=exc) from exc

    async def _fall_back(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        *,
        confidence: float | None,
    ) -> ToolCall | None:
        if self.fallback is None:
            return None
        result = await self.fallback.call(messages, tools)
        if result is None or confidence is None:
            return result
        # Jev still chose the route; only the argument filling was delegated.
        return dataclasses.replace(result, confidence=confidence)


__all__ = ["JevBackend"]
