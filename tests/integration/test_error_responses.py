from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
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
    # Relaxation note: AIRouter's _handle re-raises AIRouterError subclasses
    # (including LLMBackendError) so callers can use FastAPI exception handlers.
    # Only non-library upstream exceptions get wrapped into the 502 envelope.
    # We therefore exercise the 502 path with a generic RuntimeError, which is
    # the realistic "vendor SDK threw something unexpected" case.
    backend = make_backend(returns=RuntimeError("upstream timeout"))
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
