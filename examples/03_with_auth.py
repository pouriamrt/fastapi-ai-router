"""Both layers of auth: gate /ai itself, and rely on each route's Depends() too."""

from fastapi import Depends, FastAPI, Header, HTTPException

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.litellm import LiteLLMBackend


def require_paid_tier(x_tier: str = Header(...)) -> None:
    if x_tier != "paid":
        raise HTTPException(status_code=402)


def require_user(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401)
    return authorization.removeprefix("Bearer ")


app = FastAPI()


@app.post("/orders/{order_id}/cancel")
@ai_route(description="Cancel an order; requires authenticated user.")
def cancel(order_id: int, reason: str | None = None, user: str = Depends(require_user)) -> dict:
    return {"cancelled": order_id, "user": user, "reason": reason}


# Layer 1 = require_paid_tier (gates /ai)
# Layer 2 = require_user (gates /orders/.../cancel)
AIRouter(
    app,
    llm=LiteLLMBackend(model="gpt-4o-mini"),
    mode="decorator",
    dependencies=[Depends(require_paid_tier)],
)
