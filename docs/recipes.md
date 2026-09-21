# Recipes

## Bring your own LLM backend

Implement the `LLMBackend` Protocol:

```python
from fastapi_ai_router import LLMBackend, Message, ToolCall, ToolDef


class MyBackend:
    async def call(self, messages: list[Message], tools: list[ToolDef]) -> ToolCall | None:
        # call your LLM, parse the response, return ToolCall(...) or None
        ...
```

Pass an instance to `AIRouter(app, llm=MyBackend())`. No subclassing required.

## Forward additional headers

By default AIRouter forwards `authorization, cookie, x-api-key, x-forwarded-for, x-request-id`.
Override:

```python
from fastapi_ai_router import AIRouter, DEFAULT_FORWARD_HEADERS

AIRouter(app, llm=..., forward_headers=DEFAULT_FORWARD_HEADERS | {"x-tenant-id"})
```

## Pipe decisions to Langfuse / OpenTelemetry / Sentry

```python
async def to_langfuse(d):
    await langfuse_client.log(...)


AIRouter(app, llm=..., on_decision=to_langfuse)
```

The library has no hard dependency on any tracing vendor. The hook is async,
so any I/O is fine.

## Large apps (100+ routes)

v0.1 sends every selected tool to the LLM on every call. For apps with many
routes, prefer `mode="decorator"` so you control exactly which routes the LLM
sees. v0.2 adds semantic prefiltering for the "expose hundreds of routes" case.

## Force a registry rebuild

If you mutate `app.routes` at runtime (rare), call:

```python
router = AIRouter(app, llm=...)
# ... later ...
router.rebuild()  # next request rebuilds the tool list
```
