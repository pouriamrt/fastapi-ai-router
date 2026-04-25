import dataclasses

import pytest

from fastapi_ai_router.observability import (
    Decision,
    DecisionHook,
    ErrorEvent,
    ErrorHook,
)


def test_decision_is_frozen_with_expected_fields():
    d = Decision(
        request_id="r1",
        query="cancel order 1",
        tool_name="cancel_order",
        args={"order_id": 1},
        reasoning="user said cancel",
        model="fake",
        prompt_tokens=10,
        completion_tokens=2,
        llm_latency_ms=12,
        dispatch_latency_ms=4,
        result_status=200,
    )
    assert d.tool_name == "cancel_order"
    assert d.result_status == 200
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.tool_name = "other"  # type: ignore[misc]


def test_decision_allows_none_for_unmatched_query():
    d = Decision(
        request_id="r1",
        query="nonsense",
        tool_name=None,
        args={},
        reasoning=None,
        model="fake",
        prompt_tokens=10,
        completion_tokens=0,
        llm_latency_ms=12,
        dispatch_latency_ms=None,
        result_status=None,
    )
    assert d.tool_name is None
    assert d.dispatch_latency_ms is None


def test_error_event_carries_upstream():
    upstream = ValueError("oops")
    ev = ErrorEvent(
        request_id="r1",
        query="x",
        error_type="llm_backend_error",
        error_detail="oops",
        upstream=upstream,
    )
    assert ev.upstream is upstream


@pytest.mark.asyncio
async def test_decision_hook_is_async_callable_protocol():
    captured: list[Decision] = []

    async def my_hook(decision: Decision) -> None:
        captured.append(decision)

    hook: DecisionHook = my_hook
    sample = Decision(
        request_id="r", query="q", tool_name=None, args={}, reasoning=None,
        model="fake", prompt_tokens=0, completion_tokens=0,
        llm_latency_ms=0, dispatch_latency_ms=None, result_status=None,
    )
    await hook(sample)
    assert captured == [sample]


@pytest.mark.asyncio
async def test_error_hook_is_async_callable_protocol():
    captured: list[ErrorEvent] = []

    async def my_hook(event: ErrorEvent) -> None:
        captured.append(event)

    hook: ErrorHook = my_hook
    sample = ErrorEvent(
        request_id="r", query="q", error_type="x", error_detail="x", upstream=None,
    )
    await hook(sample)
    assert captured == [sample]
