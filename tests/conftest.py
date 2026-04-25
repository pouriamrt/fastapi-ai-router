"""Shared fixtures: a small FastAPI app and a configurable FakeLLMBackend."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.decorator import ai_route


class CancelInput(BaseModel):
    reason: str


def _require_user(authorization: str = Header(...)) -> str:
    if authorization != "Bearer good":
        raise HTTPException(status_code=401)
    return "user-1"


@pytest.fixture
def sample_app() -> FastAPI:
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    @ai_route(description="Cancel a customer's order.")
    def cancel(
        order_id: int,
        payload: CancelInput,
        user: str = Depends(_require_user),
    ) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "order_id": order_id,
            "reason": payload.reason,
            "user": user,
        }

    @app.get("/products")
    @ai_route(description="List products by category.")
    def list_products(category: str | None = None, limit: int = 20) -> dict[str, Any]:
        return {"items": [], "category": category, "limit": limit}

    @app.get("/admin/secret")
    def admin_secret() -> dict[str, Any]:
        return {"secret": "shhh"}

    @app.get("/dangerous")
    @ai_route(expose=False)
    def dangerous() -> dict[str, Any]:
        return {"removed": True}

    return app


def make_backend(returns: object) -> FakeLLMBackend:
    return FakeLLMBackend(returns=returns)
