import pytest

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.errors import LLMBackendError


@pytest.mark.asyncio
async def test_fake_returns_pre_canned_toolcall():
    expected = ToolCall(
        name="cancel_order",
        args={"order_id": 1},
        reasoning="cancel it",
        prompt_tokens=0,
        completion_tokens=0,
        model="fake",
    )
    backend = FakeLLMBackend(returns=expected)
    result = await backend.call(messages=[], tools=[])
    assert result == expected


@pytest.mark.asyncio
async def test_fake_returns_none_for_no_tool_call():
    backend = FakeLLMBackend(returns=None)
    result = await backend.call(messages=[], tools=[])
    assert result is None


@pytest.mark.asyncio
async def test_fake_raises_on_exception_returns():
    backend = FakeLLMBackend(returns=LLMBackendError("upstream broke"))
    with pytest.raises(LLMBackendError, match="upstream broke"):
        await backend.call(messages=[], tools=[])


@pytest.mark.asyncio
async def test_fake_records_calls():
    backend = FakeLLMBackend(returns=None)
    await backend.call(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    )
    assert len(backend.calls) == 1
    assert backend.calls[0].messages[0]["content"] == "hi"
