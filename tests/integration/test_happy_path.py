from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
from tests.conftest import make_backend


def test_post_to_ai_dispatches_through_loopback(sample_app):
    backend = make_backend(
        ToolCall(
            name="cancel",
            args={"order_id": 7, "reason": "duplicate"},
            reasoning="cancel it",
            prompt_tokens=10,
            completion_tokens=2,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)

    resp = client.post(
        "/ai",
        json={"query": "cancel order 7 because it's a duplicate"},
        headers={"authorization": "Bearer good"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["endpoint"] == "POST /orders/{order_id}/cancel"
    assert body["args"] == {"order_id": 7, "reason": "duplicate"}
    assert body["result"]["status"] == "cancelled"
    assert body["result_status"] == 200
