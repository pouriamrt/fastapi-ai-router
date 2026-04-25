"""Pipe Decision/ErrorEvent into your favorite observability stack.

This example logs to stdout — replace with langfuse_client.log(...) etc.
"""

from fastapi import FastAPI

from fastapi_ai_router import AIRouter, Decision, ErrorEvent, ai_route
from fastapi_ai_router.backends.litellm import LiteLLMBackend


async def log_decision(d: Decision) -> None:
    print(
        f"[router] {d.request_id} model={d.model} tool={d.tool_name} "
        f"prompt_tokens={d.prompt_tokens} latency_ms={d.llm_latency_ms} "
        f"status={d.result_status}"
    )


async def log_error(e: ErrorEvent) -> None:
    print(f"[router-error] {e.request_id} type={e.error_type} detail={e.error_detail}")


app = FastAPI()


@app.get("/products")
@ai_route(description="List products by category.")
def list_products(category: str | None = None) -> dict:
    return {"items": [], "category": category}


AIRouter(
    app,
    llm=LiteLLMBackend(model="gpt-4o-mini"),
    mode="decorator",
    on_decision=log_decision,
    on_error=log_error,
)
