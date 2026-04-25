"""Tag-based exposure: any route tagged 'ai' is reachable via /ai."""

from fastapi import FastAPI

from fastapi_ai_router import AIRouter
from fastapi_ai_router.backends.litellm import LiteLLMBackend

app = FastAPI()


@app.get("/products", tags=["ai"])
def list_products(category: str | None = None) -> dict:
    return {"items": [], "category": category}


@app.get("/orders/{id}", tags=["ai"])
def get_order(id: int) -> dict:
    return {"id": id, "status": "open"}


@app.get("/internal/metrics")  # no "ai" tag → not exposed
def metrics() -> dict:
    return {"qps": 12.5}


AIRouter(app, llm=LiteLLMBackend(model="gpt-4o-mini"), mode="tag", tag="ai")
