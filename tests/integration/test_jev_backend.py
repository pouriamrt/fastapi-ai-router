from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.backends.jev import NO_ROUTE, NOT_STATED, ROUTE_QID, JevBackend
from fastapi_ai_router.core import AIRouter
from fastapi_ai_router.observability import Decision
from tests.conftest import JevStubClient

AUTH = {"Authorization": "Bearer good"}


def test_jev_backend_routes_and_dispatches(sample_app):
    decisions: list[Decision] = []

    async def hook(d: Decision) -> None:
        decisions.append(d)

    # sample_app slots: cancel -> arg0 order_id, arg1 reason; list_products -> arg2, arg3
    stub = JevStubClient(
        {ROUTE_QID: ("cancel", 0.97), "arg0": ("7", 1.0), "arg1": ("duplicate", 0.8)}
    )
    AIRouter(sample_app, llm=JevBackend(client=stub), on_decision=hook)
    resp = TestClient(sample_app).post(
        "/ai", json={"query": "cancel order 7 as a duplicate"}, headers=AUTH
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["args"] == {"order_id": 7, "reason": "duplicate"}
    assert body["result"]["status"] == "cancelled"
    assert decisions[0].confidence == 0.97


def test_jev_backend_missing_order_id_returns_422(sample_app):
    stub = JevStubClient({ROUTE_QID: ("cancel", 0.9), "arg1": (NOT_STATED, 1.0)})
    AIRouter(sample_app, llm=JevBackend(client=stub))
    resp = TestClient(sample_app).post("/ai", json={"query": "cancel my order"}, headers=AUTH)
    assert resp.status_code == 422
    assert resp.json()["missing"] == ["order_id"]


def test_jev_backend_no_match_returns_422(sample_app):
    stub = JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})
    AIRouter(sample_app, llm=JevBackend(client=stub))
    resp = TestClient(sample_app).post("/ai", json={"query": "what's the weather"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"


def test_jev_backend_low_confidence_uses_fallback(sample_app):
    fallback = FakeLLMBackend(
        returns=ToolCall(
            name="list_products",
            args={"category": "books"},
            reasoning="llm picked",
            prompt_tokens=5,
            completion_tokens=1,
            model="llm",
        )
    )
    stub = JevStubClient({ROUTE_QID: ("cancel", 0.2)})
    AIRouter(sample_app, llm=JevBackend(client=stub, fallback=fallback))
    resp = TestClient(sample_app).post("/ai", json={"query": "books please"})
    assert resp.status_code == 200
    assert resp.json()["args"] == {"category": "books"}
