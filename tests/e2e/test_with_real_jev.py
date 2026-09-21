"""Real Jev smoke tests. Gated behind RUN_LLM_TESTS=1 and TYPESAFE_API_KEY.

Run: RUN_LLM_TESTS=1 uv run --env-file .env pytest tests/e2e/test_with_real_jev.py -v
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.jev import JevBackend

pytestmark = pytest.mark.e2e

CANCEL = "POST /orders/{order_id}/cancel"
PRODUCTS = "GET /products"


@pytest.fixture(autouse=True)
def gate() -> None:
    if os.environ.get("RUN_LLM_TESTS") != "1":
        pytest.skip("RUN_LLM_TESTS=1 not set")
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY not set")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    @ai_route(description="Cancel a customer's order and record a reason.")
    def cancel_order(order_id: int, reason: str | None = None) -> dict:
        return {"status": "cancelled", "order_id": order_id, "reason": reason}

    @app.get("/products")
    @ai_route(description="Search products by category with an optional limit.")
    def list_products(category: str | None = None, limit: int = 20) -> dict:
        return {"items": [], "category": category, "limit": limit}

    AIRouter(app, llm=JevBackend())
    return TestClient(app)


@pytest.mark.parametrize(
    ("query", "endpoint", "expected_args"),
    [
        ("cancel order 123 because it was a duplicate", CANCEL, {"order_id": 123}),
        ("cancel my order #4521, I ordered the wrong size", CANCEL, {"order_id": 4521}),
        ("cancel order 77", CANCEL, {"order_id": 77}),
        ("show me 5 laptops", PRODUCTS, {"category": "laptops", "limit": 5}),
        ("list 10 items in the kitchen category", PRODUCTS, {"category": "kitchen", "limit": 10}),
    ],
)
def test_jev_routes_and_extracts(client, query, endpoint, expected_args):
    resp = client.post("/ai", json={"query": query})
    assert resp.status_code == 200
    body = resp.json()
    assert body["endpoint"] == endpoint
    assert {k: body["args"].get(k) for k in expected_args} == expected_args


def test_jev_extracts_free_text_reason(client):
    body = client.post("/ai", json={"query": "cancel order 123 because it was a duplicate"}).json()
    assert "duplicate" in body["args"]["reason"]


def test_jev_leaves_unstated_optionals_out(client):
    body = client.post("/ai", json={"query": "show me products"}).json()
    assert body["endpoint"] == PRODUCTS
    assert body["args"] == {}


def test_jev_no_route_for_off_topic_query(client):
    resp = client.post("/ai", json={"query": "what's the weather in Paris"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"
