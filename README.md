# fastapi-ai-router

Turn your FastAPI app's existing routes into a natural-language-callable surface
using LLM function-calling. Drop-in middleware. No new metadata required.

```python
from fastapi import FastAPI
from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.litellm import LiteLLMBackend

app = FastAPI()

@app.post("/orders/{order_id}/cancel")
@ai_route(description="Cancel a customer's order.")
def cancel_order(order_id: int, reason: str | None = None):
    ...

AIRouter(app, llm=LiteLLMBackend(model="gpt-4o-mini"))
```

```bash
$ curl -X POST localhost:8000/ai -d '{"query":"cancel order 123 because duplicate"}'
{
  "endpoint": "POST /orders/123/cancel",
  "args": {"order_id": 123, "reason": "duplicate"},
  "result": {"status": "cancelled"},
  "reasoning": "User wants to cancel order 123."
}
```

## Install

```bash
pip install fastapi-ai-router[litellm]
```

The `[litellm]` extra is the simplest way to get a working backend (OpenAI,
Anthropic, Gemini, Ollama, etc., all via LiteLLM). To bring your own backend,
implement `LLMBackend` and pass it to `AIRouter(..., llm=YourBackend())`.

## How it works

1. On first request, AIRouter walks `app.routes` and projects each one into a
   JSON-Schema tool definition (using FastAPI's existing OpenAPI machinery).
2. The user's `{"query": "..."}` is sent to your LLM with those tools.
3. The LLM picks one tool with arguments. AIRouter dispatches the call
   internally via `httpx + ASGITransport` — so your existing `Depends(auth)`,
   middleware, validation, and exception handlers all run normally.
4. The dispatched response is wrapped in an envelope and returned to the client.

## Exposure modes

| `mode=` | What it exposes |
|---|---|
| `"decorator"` (default) | Only routes with `@ai_route(expose=True)`. |
| `"tag"` | Only routes whose `tags=` includes the configured tag. |
| `"all"` | Every route, minus `exclude=` patterns and routes marked `@ai_route(expose=False)`. |

There is no silent fallback between modes. Pick one explicitly. Default is
`decorator` (safest).

## Limitations (v0.1)

This library deliberately does not do, in v0.1:

- Multi-step / agent-loop reasoning (single-shot only).
- Conversation history or session state.
- Semantic caching.
- Streaming responses.
- Form/multipart bodies in dispatched routes.

These are v0.2+ candidates. See [docs/concepts.md](docs/concepts.md).

## Documentation

- [Concepts](docs/concepts.md) — mental model, request flow, two-layer auth.
- [Recipes](docs/recipes.md) — custom backend, tracing integrations, large-app strategies.
- [Security](docs/security.md) — when `mode="all"` is dangerous, prompt-injection considerations.

## License

MIT.
