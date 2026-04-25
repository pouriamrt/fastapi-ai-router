from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
from fastapi_ai_router.errors import LLMBackendError
from tests.conftest import make_backend


def test_no_route_matched_returns_422(sample_app):
    backend = make_backend(returns=None)
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "do something irrelevant"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"


def test_unknown_tool_returns_422(sample_app):
    backend = make_backend(
        ToolCall(
            name="not_a_real_tool",
            args={},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "do thing"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "unknown_tool"
    assert body["tool_name"] == "not_a_real_tool"


def test_llm_backend_error_returns_502(sample_app):
    backend = make_backend(returns=LLMBackendError("upstream timeout"))
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "anything"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "llm_backend_error"
    assert "upstream timeout" in body["detail"]


def test_dispatched_route_4xx_passes_through_status(sample_app):
    backend = make_backend(
        ToolCall(
            name="cancel",
            args={"order_id": 1, "reason": "x"},
            reasoning="cancel",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    # missing authorization → cancel's auth dep fails (401 or 422)
    resp = client.post("/ai", json={"query": "cancel"})
    assert resp.status_code in (401, 422)
    assert resp.json().get("result_status") in (401, 422)


def test_dispatch_error_returns_500(sample_app, monkeypatch):
    """If the loopback dispatcher itself fails (transport-level error), the
    library returns 500 with a structured envelope rather than letting the
    exception propagate."""
    from fastapi_ai_router import core as core_mod

    backend = make_backend(
        ToolCall(
            name="list_products",
            args={"category": "books"},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")

    async def boom(**kwargs):
        raise RuntimeError("transport exploded")

    monkeypatch.setattr(core_mod, "dispatch", boom)

    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "list books"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "dispatch_error"
    assert "transport exploded" in body["detail"]
