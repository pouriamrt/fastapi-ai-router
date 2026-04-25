from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.core import AIRouter
from tests.conftest import make_backend


def test_decorator_mode_excludes_undecorated_routes(sample_app):
    backend = make_backend(
        ToolCall(
            name="admin_secret",
            args={},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "tell me the secret"})
    body = resp.json()
    assert resp.status_code == 422
    assert body["error"] == "unknown_tool"
    assert "admin_secret" not in body["available_tools"]


def test_all_mode_excludes_kill_switch_and_excluded_paths(sample_app):
    backend = make_backend(
        ToolCall(
            name="dangerous",
            args={},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="all", exclude=["/admin/*"])
    client = TestClient(sample_app)
    resp = client.post("/ai", json={"query": "do dangerous"})
    body = resp.json()
    assert resp.status_code == 422
    assert body["error"] == "unknown_tool"
    assert "dangerous" not in body["available_tools"]
    assert "admin_secret" not in body["available_tools"]


def test_raw_query_bypasses_envelope(sample_app):
    backend = make_backend(
        ToolCall(
            name="list_products",
            args={"category": "books", "limit": 5},
            reasoning="",
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator")
    client = TestClient(sample_app)
    resp = client.post("/ai?raw=true", json={"query": "list books"})
    assert resp.status_code == 200
    body = resp.json()
    # raw mode returns the dispatched body directly — no envelope keys
    assert "endpoint" not in body
    assert body == {"items": [], "category": "books", "limit": 5}
