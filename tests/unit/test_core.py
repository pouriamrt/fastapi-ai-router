import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.core import AIRouter
from fastapi_ai_router.decorator import ai_route


def test_constructor_registers_endpoint():
    app = FastAPI()
    backend = FakeLLMBackend(returns=None)
    AIRouter(app, llm=backend)
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/ai" in paths


def test_constructor_respects_custom_endpoint():
    app = FastAPI()
    AIRouter(app, llm=FakeLLMBackend(returns=None), endpoint="/copilot")
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/copilot" in paths


def test_invalid_mode_raises():
    app = FastAPI()
    with pytest.raises(ValueError):
        AIRouter(app, llm=FakeLLMBackend(returns=None), mode="bogus")  # type: ignore[arg-type]


def test_basic_request_flow_returns_envelope():
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    @ai_route(description="Cancel an order.")
    def cancel(order_id: int, reason: str | None = None) -> dict:
        return {"status": "cancelled", "order_id": order_id, "reason": reason}

    backend = FakeLLMBackend(
        returns=ToolCall(
            name="cancel",
            args={"order_id": 7, "reason": "duplicate"},
            reasoning="user wants cancel",
            prompt_tokens=10,
            completion_tokens=2,
            model="fake",
        )
    )
    AIRouter(app, llm=backend, mode="decorator")

    client = TestClient(app)
    resp = client.post("/ai", json={"query": "cancel order 7 because it's a duplicate"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["endpoint"] == "POST /orders/{order_id}/cancel"
    assert body["args"] == {"order_id": 7, "reason": "duplicate"}
    # Note: in this route, FastAPI classifies `reason` as query (not body), so it
    # arrives correctly via query string from the dispatcher.
    assert body["result"] == {"status": "cancelled", "order_id": 7, "reason": "duplicate"}
    assert body["reasoning"] == "user wants cancel"
    assert body["result_status"] == 200


def test_rebuild_invalidates_cache():
    app = FastAPI()

    @app.get("/a")
    @ai_route()
    def a() -> dict:
        return {"a": 1}

    router = AIRouter(app, llm=FakeLLMBackend(returns=None))

    client = TestClient(app)
    # First call builds the registry
    client.post("/ai", json={"query": "x"})

    # Add a new route after first build
    @app.get("/b")
    @ai_route()
    def b() -> dict:
        return {"b": 2}

    # Without rebuild, b is not registered
    assert "b" not in router._registry  # type: ignore[attr-defined]

    router.rebuild()
    # Force build by triggering a request
    client.post("/ai", json={"query": "x"})
    assert "b" in router._registry  # type: ignore[attr-defined]
