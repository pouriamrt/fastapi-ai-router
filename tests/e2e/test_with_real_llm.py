"""Real-LLM smoke tests. Gated behind RUN_LLM_TESTS=1.

Run on release tags only — they cost money and require an API key.
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.litellm import LiteLLMBackend

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def gate() -> None:
    if os.environ.get("RUN_LLM_TESTS") != "1":
        pytest.skip("RUN_LLM_TESTS=1 not set")


def test_real_llm_picks_obvious_route():
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    @ai_route(description="Cancel a customer's order.")
    def cancel(order_id: int, reason: str | None = None) -> dict:
        return {"status": "cancelled", "order_id": order_id, "reason": reason}

    @app.get("/products")
    @ai_route(description="List products.")
    def list_products(category: str | None = None) -> dict:
        return {"items": [], "category": category}

    AIRouter(app, llm=LiteLLMBackend(model="gpt-4o-mini"))
    client = TestClient(app)
    resp = client.post(
        "/ai",
        json={"query": "please cancel order 42 because the customer changed their mind"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "cancel" in body["endpoint"].lower()
    assert body["args"]["order_id"] == 42


def test_real_llm_returns_no_route_for_irrelevant_query():
    app = FastAPI()

    @app.get("/products")
    @ai_route(description="List products.")
    def list_products() -> dict:
        return {"items": []}

    AIRouter(app, llm=LiteLLMBackend(model="gpt-4o-mini"))
    client = TestClient(app)
    resp = client.post("/ai", json={"query": "what is the meaning of life?"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"
