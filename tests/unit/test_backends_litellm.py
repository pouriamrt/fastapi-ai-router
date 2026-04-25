from unittest.mock import AsyncMock, patch

import pytest

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.litellm import LiteLLMBackend


@pytest.mark.asyncio
async def test_litellm_translates_tool_call_response():
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": "I'll cancel order 7.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "cancel_order",
                                "arguments": '{"order_id": 7, "reason": "duplicate"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 14, "completion_tokens": 5},
        "model": "gpt-5-mini",
    }
    backend = LiteLLMBackend(model="gpt-5-mini")
    with patch(
        "fastapi_ai_router.backends.litellm.acompletion",
        new=AsyncMock(return_value=fake_response),
    ):
        result = await backend.call(
            messages=[{"role": "user", "content": "cancel order 7"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "cancel_order",
                        "description": "Cancel an order.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

    assert isinstance(result, ToolCall)
    assert result.name == "cancel_order"
    assert result.args == {"order_id": 7, "reason": "duplicate"}
    assert result.reasoning == "I'll cancel order 7."
    assert result.prompt_tokens == 14
    assert result.completion_tokens == 5
    assert result.model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_litellm_returns_none_when_no_tool_call():
    fake_response = {
        "choices": [{"message": {"content": "I don't know.", "tool_calls": None}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        "model": "gpt-5-mini",
    }
    backend = LiteLLMBackend(model="gpt-5-mini")
    with patch(
        "fastapi_ai_router.backends.litellm.acompletion",
        new=AsyncMock(return_value=fake_response),
    ):
        result = await backend.call(messages=[], tools=[])
    assert result is None


@pytest.mark.asyncio
async def test_litellm_wraps_upstream_exception():
    from fastapi_ai_router.errors import LLMBackendError

    backend = LiteLLMBackend(model="gpt-5-mini")
    with patch(
        "fastapi_ai_router.backends.litellm.acompletion",
        new=AsyncMock(side_effect=RuntimeError("rate limited")),
    ):
        with pytest.raises(LLMBackendError) as ei:
            await backend.call(messages=[], tools=[])
        assert isinstance(ei.value.upstream, RuntimeError)
