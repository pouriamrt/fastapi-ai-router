# Examples

Each example is runnable with `uvicorn`. Set the appropriate API key env var
for whatever model your `LiteLLMBackend` is pointed at.

| File | Demonstrates |
|---|---|
| `01_basic.py` | The smallest possible app — single `AIRouter(app, llm=...)` line. |
| `02_tag_mode.py` | Tag-based exposure (`mode="tag"`). |
| `03_with_auth.py` | Two-layer auth: `dependencies=` on /ai + per-route Depends(). |
| `04_with_observability.py` | `on_decision` / `on_error` hooks for tracing. |
| `05_jev.py` | Routing with TypeSafe's Jev classifier instead of an LLM (needs `TYPESAFE_API_KEY`). |
