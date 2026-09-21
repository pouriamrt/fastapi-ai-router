import asyncio
import time

import pytest
from typesafe_sdk import TypeSafeError

from fastapi_ai_router.backends import ToolCall, jev
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.backends.jev import (
    MAX_CHOICE_OPTIONS,
    NO_ROUTE,
    NOT_STATED,
    ROUTE_QID,
    ArgSlot,
    JevBackend,
    build_questions,
    build_slots,
    candidates,
    classify,
    coerce,
)
from fastapi_ai_router.errors import LLMBackendError
from tests.conftest import JevStubClient

CANCEL_TOOL = {
    "type": "function",
    "function": {
        "name": "cancel_order",
        "description": "Cancel a customer's order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["order_id"],
        },
    },
}
PRODUCTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_products",
        "description": "Search products by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
    },
}
TAGS_TOOL = {
    "type": "function",
    "function": {
        "name": "tag_item",
        "description": "Tag an item.",
        "parameters": {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    },
}
TOOLS = [CANCEL_TOOL, PRODUCTS_TOOL]


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "integer"}, ("integer", ())),
        ({"type": "number"}, ("number", ())),
        ({"type": "boolean"}, ("boolean", ())),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, ("string", ())),
        ({"enum": ["s", "m"], "type": "string"}, ("enum", ("s", "m"))),
        ({"type": "array", "items": {"type": "string"}}, (None, ())),
        ({"type": "object"}, (None, ())),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, (None, ())),
    ],
)
def test_classify_shapes(schema, expected):
    assert classify(schema, {}) == expected


def test_classify_resolves_local_and_root_defs():
    color = {"enum": ["red", "blue"], "type": "string"}
    local = {
        "$defs": {"Color": color},
        "anyOf": [{"$ref": "#/$defs/Color"}, {"type": "null"}],
    }
    assert classify(local, {}) == ("enum", ("red", "blue"))
    assert classify({"$ref": "#/$defs/Color"}, {"Color": color}) == ("enum", ("red", "blue"))


def test_classify_dangling_ref_is_unsupported():
    assert classify({"$ref": "#/$defs/Missing"}, {}) == (None, ())


def test_candidates_integer_strips_punctuation():
    assert candidates("cancel my order #4521, now", "integer", (), 6) == ("4521",)


def test_candidates_number_includes_decimals():
    assert candidates("under 19.99 or 20", "number", (), 6) == ("19.99", "20")


def test_candidates_string_ngrams_are_bounded_and_deduplicated():
    assert candidates("red red shoe", "string", (), 2) == ("red", "red red", "red shoe", "shoe")


def test_candidates_enum_and_boolean():
    assert candidates("anything", "enum", ("s", 2), 6) == ("s", "2")
    assert candidates("anything", "boolean", (), 6) == ("true", "false")


def test_candidates_cap_leaves_room_for_not_stated():
    query = " ".join(f"w{i}" for i in range(100))
    assert len(candidates(query, "string", (), 6)) == MAX_CHOICE_OPTIONS - 1


def test_candidates_drop_literal_not_stated():
    assert NOT_STATED not in candidates("reason not_stated", "string", (), 1)


def test_candidates_integer_keeps_comma_thousands_grouping():
    assert candidates("transfer 1,000 dollars", "integer", (), 6) == ("1,000",)


def test_candidates_number_keeps_comma_grouping_and_decimals():
    assert candidates("total $1,299.99", "number", (), 6) == ("1,299.99",)


def test_candidates_integer_keeps_leading_sign():
    assert candidates("balance -5", "integer", (), 6) == ("-5",)


def test_candidates_integer_dash_in_a_word_is_not_a_sign():
    assert candidates("order-123", "integer", (), 6) == ("123",)


def test_candidates_integer_dash_between_numbers_is_not_a_sign():
    assert candidates("pick 5-10", "integer", (), 6) == ("5", "10")


def test_candidates_number_leading_dot_decimal():
    assert candidates("weight .5", "number", (), 6) == (".5",)


def test_coerce_integer_strips_comma_grouping():
    assert coerce(_slot("integer"), "1,000") == 1000


def test_coerce_number_strips_comma_grouping():
    assert coerce(_slot("number"), "1,299.99") == 1299.99


def test_coerce_number_leading_dot_decimal():
    assert coerce(_slot("number"), ".5") == 0.5


def test_build_slots_covers_every_route_param():
    slots = build_slots("cancel order 123", TOOLS, 6)
    assert [(s.qid, s.route, s.param, s.kind, s.required) for s in slots] == [
        ("arg0", "cancel_order", "order_id", "integer", True),
        ("arg1", "cancel_order", "reason", "string", False),
        ("arg2", "list_products", "category", "string", False),
        ("arg3", "list_products", "limit", "integer", False),
    ]
    assert slots[0].options == ("123",)


def test_build_slots_bounds_and_shares_string_candidate_generation():
    query = " ".join(f"w{i}" for i in range(20_000))
    tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search.",
            "parameters": {
                "type": "object",
                "properties": {f"p{i}": {"type": "string"} for i in range(20)},
                "required": [],
            },
        },
    }
    start = time.perf_counter()
    slots = build_slots(query, [tool], 6)
    elapsed = time.perf_counter() - start
    assert len(slots) == 20
    assert all(len(slot.options) == MAX_CHOICE_OPTIONS - 1 for slot in slots)
    assert elapsed < 0.3


def test_build_slots_marks_unsupported_types():
    slots = build_slots("tag it red", [TAGS_TOOL], 6)
    assert (slots[0].kind, slots[0].options, slots[0].required) == (None, (), True)


def test_build_questions_asks_route_and_slots_with_candidates():
    slots = build_slots("show products", [PRODUCTS_TOOL], 6)
    questions = build_questions([PRODUCTS_TOOL], slots)
    # `limit` has no digits to choose from, so it gets no question.
    assert set(questions) == {ROUTE_QID, "arg0"}
    assert set(questions[ROUTE_QID].criteria) == {"list_products", NO_ROUTE}
    assert NOT_STATED in questions["arg0"].criteria


def test_build_questions_skips_unsupported_types():
    slots = build_slots("tag it red", [TAGS_TOOL], 6)
    assert set(build_questions([TAGS_TOOL], slots)) == {ROUTE_QID}


def _slot(kind, enum_values=()):
    return ArgSlot(
        qid="arg0",
        route="r",
        param="p",
        kind=kind,
        required=False,
        options=(),
        enum_values=enum_values,
    )


@pytest.mark.parametrize(
    ("kind", "raw", "expected"),
    [
        ("integer", "4521", 4521),
        ("integer", "4.5", None),
        ("number", "19.99", 19.99),
        ("boolean", "true", True),
        ("boolean", "false", False),
        ("string", "wrong size", "wrong size"),
    ],
)
def test_coerce(kind, raw, expected):
    assert coerce(_slot(kind), raw) == expected


def test_coerce_enum_maps_back_to_original_value():
    assert coerce(_slot("enum", (1, 2)), "2") == 2
    assert coerce(_slot("enum", ("s",)), "xl") is None


def _messages(query):
    return [
        {"role": "system", "content": "You are a router."},
        {"role": "user", "content": query},
    ]


async def test_call_routes_and_extracts_args_in_one_request():
    stub = JevStubClient(
        {
            ROUTE_QID: ("cancel_order", 0.98),
            "arg0": ("123", 1.0),
            "arg1": ("because it was a duplicate", 0.6),
        }
    )
    query = "cancel order 123 because it was a duplicate"
    result = await JevBackend(client=stub).call(_messages(query), TOOLS)
    assert result == ToolCall(
        name="cancel_order",
        args={"order_id": 123, "reason": "because it was a duplicate"},
        reasoning=None,
        prompt_tokens=100,
        completion_tokens=10,
        model="jev-1.13.0",
        confidence=0.98,
    )
    assert len(stub.calls) == 1
    state, questions = stub.calls[0]
    assert state == query
    assert set(questions) == {ROUTE_QID, "arg0", "arg1", "arg2", "arg3"}


async def test_call_omits_optional_not_stated_args():
    stub = JevStubClient(
        {ROUTE_QID: ("cancel_order", 1.0), "arg0": ("77", 1.0), "arg1": (NOT_STATED, 1.0)}
    )
    result = await JevBackend(client=stub).call(_messages("cancel order 77"), TOOLS)
    assert result is not None
    assert result.args == {"order_id": 77}


async def test_confident_no_match_returns_none_without_calling_fallback():
    fallback = FakeLLMBackend(returns=None)
    stub = JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})
    backend = JevBackend(client=stub, fallback=fallback)
    assert await backend.call(_messages("weather in Paris"), TOOLS) is None
    assert fallback.calls == []


async def test_low_confidence_without_fallback_returns_none():
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.3)})
    assert await JevBackend(client=stub).call(_messages("hmm"), TOOLS) is None


async def test_low_confidence_hands_all_tools_to_fallback():
    llm_call = ToolCall(
        name="list_products",
        args={},
        reasoning="r",
        prompt_tokens=5,
        completion_tokens=1,
        model="llm",
    )
    fallback = FakeLLMBackend(returns=llm_call)
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.3)})
    result = await JevBackend(client=stub, fallback=fallback).call(_messages("hmm"), TOOLS)
    assert result == llm_call
    assert fallback.calls[0].tools == TOOLS


async def test_missing_required_arg_narrows_fallback_to_chosen_route():
    llm_call = ToolCall(
        name="cancel_order",
        args={"order_id": 5},
        reasoning=None,
        prompt_tokens=5,
        completion_tokens=1,
        model="llm",
    )
    fallback = FakeLLMBackend(returns=llm_call)
    # "cancel my order" has no digits, so order_id gets no question at all.
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9), "arg1": (NOT_STATED, 1.0)})
    backend = JevBackend(client=stub, fallback=fallback)
    result = await backend.call(_messages("cancel my order"), TOOLS)
    assert fallback.calls[0].tools == [CANCEL_TOOL]
    assert result is not None
    assert (result.model, result.confidence) == ("llm", 0.9)


async def test_missing_required_arg_without_fallback_returns_partial_call():
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9), "arg1": (NOT_STATED, 1.0)})
    result = await JevBackend(client=stub).call(_messages("cancel my order"), TOOLS)
    assert result is not None
    assert (result.name, result.args) == ("cancel_order", {})


async def test_unsupported_required_param_goes_to_fallback():
    fallback = FakeLLMBackend(returns=None)
    stub = JevStubClient({ROUTE_QID: ("tag_item", 0.95)})
    await JevBackend(client=stub, fallback=fallback).call(_messages("tag it red"), [TAGS_TOOL])
    assert fallback.calls[0].tools == [TAGS_TOOL]


async def test_sdk_errors_become_llm_backend_errors():
    stub = JevStubClient(TypeSafeError("rate limited"))
    with pytest.raises(LLMBackendError) as ei:
        await JevBackend(client=stub).call(_messages("cancel order 1"), TOOLS)
    assert isinstance(ei.value.upstream, TypeSafeError)


async def test_missing_expected_answer_raises():
    # order_id had a candidate ("1"), so an answer for arg0 was expected.
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9)})
    with pytest.raises(LLMBackendError, match="arg0"):
        await JevBackend(client=stub).call(_messages("cancel order 1"), TOOLS)


async def test_client_is_created_once_from_settings(monkeypatch):
    made = []

    def factory(**kwargs):
        made.append(kwargs)
        return JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})

    monkeypatch.setattr(jev, "AsyncTypeSafeClient", factory)
    backend = JevBackend(api_key="k", model="jev-1.13.0")
    await backend.call(_messages("hi"), TOOLS)
    await backend.call(_messages("hi"), TOOLS)
    assert made == [{"api_key": "k", "model": "jev-1.13.0"}]


def test_api_key_does_not_appear_in_repr():
    assert "SECRET-XYZ" not in repr(JevBackend(api_key="SECRET-XYZ"))


def test_client_rebuilt_when_event_loop_changes(monkeypatch):
    made = []

    def factory(**kwargs):
        made.append(kwargs)
        return JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})

    monkeypatch.setattr(jev, "AsyncTypeSafeClient", factory)
    backend = JevBackend(api_key="k")
    asyncio.run(backend.call(_messages("hi"), TOOLS))
    asyncio.run(backend.call(_messages("hi"), TOOLS))
    assert len(made) == 2


def test_injected_client_is_never_replaced_across_loops():
    stub = JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})
    backend = JevBackend(client=stub)
    asyncio.run(backend.call(_messages("hi"), TOOLS))
    asyncio.run(backend.call(_messages("hi"), TOOLS))
    assert backend.client is stub
    assert len(stub.calls) == 2
