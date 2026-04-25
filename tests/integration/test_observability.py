from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
from fastapi_ai_router.observability import Decision, ErrorEvent
from tests.conftest import make_backend


def test_on_decision_fires_with_full_payload(sample_app):
    captured: list[Decision] = []

    async def hook(d: Decision) -> None:
        captured.append(d)

    backend = make_backend(
        ToolCall(
            name="list_products",
            args={"category": "books"},
            reasoning="user wants books",
            prompt_tokens=12,
            completion_tokens=3,
            model="fake-model",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator", on_decision=hook)
    client = TestClient(sample_app)
    client.post("/ai", json={"query": "list books"})

    assert len(captured) == 1
    d = captured[0]
    assert d.tool_name == "list_products"
    assert d.args == {"category": "books"}
    assert d.reasoning == "user wants books"
    assert d.model == "fake-model"
    assert d.prompt_tokens == 12
    assert d.completion_tokens == 3
    assert d.result_status == 200


def test_on_error_fires_for_no_route_matched(sample_app):
    captured: list[ErrorEvent] = []

    async def hook(e: ErrorEvent) -> None:
        captured.append(e)

    backend = make_backend(returns=None)
    AIRouter(sample_app, llm=backend, mode="decorator", on_error=hook)
    client = TestClient(sample_app)
    client.post("/ai", json={"query": "irrelevant"})

    assert len(captured) == 1
    assert captured[0].error_type == "no_route_matched"


def test_on_error_fires_for_unknown_tool(sample_app):
    captured: list[ErrorEvent] = []

    async def hook(e: ErrorEvent) -> None:
        captured.append(e)

    backend = make_backend(
        ToolCall(
            name="hallucinated_tool",
            args={},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator", on_error=hook)
    client = TestClient(sample_app)
    client.post("/ai", json={"query": "x"})
    assert len(captured) == 1
    assert captured[0].error_type == "unknown_tool"
