from fastapi import Depends, Header, HTTPException
from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
from tests.conftest import make_backend


def _require_admin(x_admin: str = Header(...)) -> None:
    if x_admin != "yes":
        raise HTTPException(status_code=401)


def test_layer1_blocks_request_before_llm_call(sample_app):
    backend = make_backend(
        ToolCall(
            name="cancel",
            args={},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(
        sample_app,
        llm=backend,
        mode="decorator",
        dependencies=[Depends(_require_admin)],
    )
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "cancel order 7"})
    # Layer 1 rejects (missing/invalid x-admin header) — could be 401 or 422
    assert resp.status_code in (401, 422)
    assert len(backend.calls) == 0


def test_layer2_rejects_dispatched_call_with_status_passthrough(sample_app):
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
    AIRouter(
        sample_app,
        llm=backend,
        mode="decorator",
        forward_headers=frozenset({"authorization"}),
    )
    client = TestClient(sample_app)
    # No bearer header → layer-2 fails inside cancel's _require_user dependency
    resp = client.post("/ai", json={"query": "cancel order 1"})
    # Could be 401 (auth header missing) or 422 (validation), depending on FastAPI version.
    assert resp.status_code in (401, 422)
    body = resp.json()
    # envelope mode: the body wraps the dispatched failure
    assert body.get("result_status") in (401, 422)


def test_layer2_passes_when_authorization_forwarded(sample_app):
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
    resp = client.post(
        "/ai",
        json={"query": "cancel order 1"},
        headers={"authorization": "Bearer good"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_status"] == 200
    assert body["result"]["user"] == "user-1"
