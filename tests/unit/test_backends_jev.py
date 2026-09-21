import pytest

from fastapi_ai_router.backends.jev import (
    MAX_CHOICE_OPTIONS,
    NO_ROUTE,
    NOT_STATED,
    ROUTE_QID,
    ArgSlot,
    build_questions,
    build_slots,
    candidates,
    classify,
    coerce,
)

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


def test_build_slots_covers_every_route_param():
    slots = build_slots("cancel order 123", TOOLS, 6)
    assert [(s.qid, s.route, s.param, s.kind, s.required) for s in slots] == [
        ("arg0", "cancel_order", "order_id", "integer", True),
        ("arg1", "cancel_order", "reason", "string", False),
        ("arg2", "list_products", "category", "string", False),
        ("arg3", "list_products", "limit", "integer", False),
    ]
    assert slots[0].options == ("123",)


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
