"""The basic example, routed by TypeSafe's Jev instead of an LLM.

Run:
    uv run --env-file .env uvicorn examples.05_jev:app --reload

Then:
    curl -X POST localhost:8000/ai -H "content-type: application/json" \
         -d '{"query":"cancel order 123 because it was a duplicate"}'
"""

from fastapi import FastAPI

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.jev import JevBackend

app = FastAPI()


@app.post("/orders/{order_id}/cancel")
@ai_route(description="Cancel a customer's order and record a reason.")
def cancel_order(order_id: int, reason: str | None = None) -> dict:
    return {"status": "cancelled", "order_id": order_id, "reason": reason}


@app.get("/products")
@ai_route(description="Search products by category with an optional limit.")
def list_products(category: str | None = None, limit: int = 20) -> dict:
    return {"items": [], "category": category, "limit": limit}


# Reads TYPESAFE_API_KEY. Pass fallback=LiteLLMBackend(...) to cover what Jev can't extract.
AIRouter(app, llm=JevBackend())
