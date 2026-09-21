import dataclasses

import pytest

from fastapi_ai_router.backends import (
    FunctionDef,
    LLMBackend,
    Message,
    ToolCall,
    ToolDef,
)


def test_toolcall_is_frozen_dataclass():
    tc = ToolCall(
        name="cancel_order",
        args={"order_id": 123},
        reasoning="user wants to cancel",
        prompt_tokens=42,
        completion_tokens=8,
        model="gpt-5-mini",
    )
    assert dataclasses.is_dataclass(tc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.name = "other"  # type: ignore[misc]


def test_message_typed_dict_keys():
    msg: Message = {"role": "user", "content": "hello"}
    assert msg["role"] == "user"
    assert msg["content"] == "hello"


def test_tool_def_shape():
    td: ToolDef = {
        "type": "function",
        "function": FunctionDef(
            name="cancel_order",
            description="Cancel an order.",
            parameters={"type": "object", "properties": {}},
        ),
    }
    assert td["type"] == "function"
    assert td["function"]["name"] == "cancel_order"


def test_llm_backend_protocol_signature():
    import inspect

    sig = inspect.signature(LLMBackend.call)
    assert {"self", "messages", "tools"} <= set(sig.parameters)
