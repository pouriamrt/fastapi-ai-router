# fastapi-ai-router — Design Specification

**Date:** 2026-04-24
**Status:** Draft, pending implementation plan
**Author:** Pouria
**License intent:** MIT (open source)

---

## 1. Summary

`fastapi-ai-router` is an open-source FastAPI middleware that turns an existing FastAPI app's routes into a natural-language-callable surface using LLM function-calling. The library reflects over `app.routes`, projects each route's OpenAPI operation into a JSON-Schema tool definition, and exposes a single `/ai` endpoint that accepts a natural-language query, asks an LLM which route to call with which arguments, and dispatches the call internally via FastAPI's own request lifecycle.

**One-line pitch:** *"Add `AIRouter(app)` to your FastAPI app. POST a natural-language query to `/ai`, the LLM picks the right endpoint, fills the args, and the middleware dispatches the call — using the OpenAPI schema FastAPI already generates."*

---

## 2. Goals

- Drop-in middleware: a single `AIRouter(app)` line enables the feature on any FastAPI app.
- Zero new metadata required: works with the OpenAPI schema FastAPI already generates.
- Bring-your-own-LLM via a `Protocol`-based interface, with a default `LiteLLMBackend` for convenience and a `FakeLLMBackend` for testing.
- Three explicit exposure modes (`decorator` / `tag` / `all`) with a safe-by-default `decorator` mode.
- Clean dispatch via httpx + ASGITransport so existing auth, validation, middleware, and exception handling all "just work."
- Pluggable observability via async hooks; no vendor dependencies in the core.
- 80%+ test coverage with a deterministic, network-free test suite.

## 3. Non-goals (v0.1)

Explicitly out of scope to keep the lane sharp:

- Multi-step / agent-loop reasoning (single-shot only).
- Conversation history or session state.
- Semantic caching of natural-language queries.
- Cost/quality routing across multiple models.
- Rate limiting beyond what existing FastAPI/Starlette middleware provides.
- Streaming responses.
- Mounting as a sub-app at arbitrary paths (single dedicated endpoint only).

These are explicit v0.2+ candidates, listed in §9.

---

## 4. Developer experience (the README story)

```python
from fastapi import FastAPI, Depends
from fastapi_ai_router import AIRouter, ai_route

app = FastAPI()

@app.post("/orders/{order_id}/cancel")
@ai_route(description="Cancel a customer's order and refund the payment.")
def cancel_order(order_id: int, reason: str | None = None, user=Depends(auth)):
    ...

@app.get("/products")
@ai_route(description="Search products by category.")
def list_products(category: str | None = None, limit: int = 20):
    ...

from fastapi_ai_router.backends import LiteLLMBackend
AIRouter(app, llm=LiteLLMBackend(model="gpt-5-mini"))  # one line to enable
```

(`llm=` is a required argument — no default. The core has no LLM-vendor dependencies; users explicitly pick a backend. `LiteLLMBackend` is a thin convenience adapter behind an extra: `pip install fastapi-ai-router[litellm]`.)

```bash
$ curl -X POST http://localhost:8000/ai \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"query": "cancel order 123 because it was a duplicate"}'
```

```json
{
  "endpoint": "POST /orders/123/cancel",
  "args": {"order_id": 123, "reason": "duplicate"},
  "result": {"status": "cancelled", "refunded": "$49.99"},
  "reasoning": "User wants to cancel order 123 with reason 'duplicate'."
}
```

### 4.1 Public configuration surface

```python
AIRouter(
    app,
    # exposure control
    mode="decorator",                  # "decorator" | "tag" | "all"  (default safe)
    tag="ai",                          # used when mode="tag"
    exclude=["/admin/*"],              # used when mode="all"

    # endpoint
    endpoint="/ai",                    # path of the AI router endpoint itself
    dependencies=[],                   # FastAPI Depends() applied to /ai (Layer-1 auth)

    # LLM backend (REQUIRED — no default)
    llm=LiteLLMBackend(model="gpt-5-mini"),  # any LLMBackend protocol implementer

    # request/response shaping
    raw_query_param="raw",             # client passes ?raw=true for pass-through response
    forward_headers=DEFAULT_FORWARD_HEADERS,  # which headers loopback to dispatched call

    # observability
    on_decision=None,                  # async hook: (Decision) -> None
    on_error=None,                     # async hook: (ErrorEvent) -> None
    debug=False,                       # logs full prompt/response for development
)
```

### 4.2 The `@ai_route` decorator

```python
def ai_route(*, description: str | None = None, expose: bool = True):
    """Marks a FastAPI route as AI-callable in mode='decorator'.
    Stores metadata on the function; does NOT modify call behavior."""
```

The decorator only attaches `__ai_route__` metadata. The route remains a normal FastAPI route in all other respects.

`expose=False` is a **kill switch** that excludes the route from AI exposure in *any* mode (including `tag` and `all`). This lets devs explicitly mark sensitive routes as never-AI-callable without relying on path patterns.

### 4.3 Exposure modes — explicit, no silent fallback

| Mode | Selects |
|---|---|
| `decorator` (default) | only routes carrying `@ai_route(expose=True)` |
| `tag` | routes whose FastAPI `tags=` list includes the configured `tag` (default `"ai"`) |
| `all` | every route, minus paths matching any pattern in `exclude` |

The mode is a single explicit constructor argument. There is no cascading fallback between modes. `mode="all"` is a deliberate, security-relevant choice and is documented as such in `docs/security.md`.

Routes with FastAPI's `include_in_schema=False` are always excluded from AI exposure regardless of mode.

---

## 5. Architecture

### 5.1 Module map

```
fastapi_ai_router/
├── __init__.py            # public exports: AIRouter, ai_route, LLMBackend, ToolCall
├── core.py                # AIRouter class — wires everything together
├── decorator.py           # @ai_route
├── introspection.py       # walk app.routes → list[RouteSpec], cached
├── schema.py              # RouteSpec; OpenAPI → tool definition
├── dispatcher.py          # un-flatten args → loopback HTTP call → response
├── envelope.py            # wrap dispatched result; respect ?raw=true
├── errors.py              # AIRouterError hierarchy
├── observability.py       # Decision, ErrorEvent, hook protocols
└── backends/
    ├── __init__.py        # LLMBackend Protocol; ToolCall dataclass
    ├── litellm.py         # default convenience adapter
    └── fake.py            # FakeLLMBackend for tests
```

Each module has one responsibility. Target file sizes 100-300 LOC.

### 5.2 Request flow (single-shot)

1. Client `POST /ai` with `{"query": "..."}` and any auth headers.
2. `AIRouter._handle()` runs (this is the FastAPI route registered at `endpoint`). Layer-1 dependencies fire here.
3. `tools = introspection.get_tools(app, mode_config)` — cached after first build (see §5.5).
4. `tool_call_or_none = await llm.call(messages=[{role:"user", content:query}], tools=tools)` returns either a `ToolCall(name, args, reasoning, ...)` or `None`.
5. **Branching:**
   - If `tool_call_or_none is None` (LLM returned plain text without invoking a tool) → raise `NoRouteMatched`.
   - If the backend returns multiple parallel tool calls → take the first; log a warning. Single-shot is the v0.1 policy. Backends should aim to return at most one.
   - Otherwise: `spec = registry.get(tool_call.name)`. If missing → raise `UnknownTool`.
6. `dispatcher.dispatch(spec, tool_call.args, request)`:
   - Splits args into path / query / body locations by `spec.param_locations`. (Form/multipart bodies are out of scope per §12.3 — routes requiring them are excluded at introspection time.)
   - Forwards configured headers per §6.2 (`authorization`, `cookie`, `x-api-key`, `x-forwarded-for`, `x-request-id` by default) to preserve auth and tracing.
   - Issues an `httpx.AsyncClient(transport=ASGITransport(app=request.app)).request(...)` call.
   - The dispatched call runs through FastAPI's full pipeline: middleware, Depends() including auth, Pydantic validation, the route function, response serialization, and exception handlers.
7. `envelope.wrap(decision, raw_response)` returns `{"endpoint", "args", "result", "reasoning", "result_status"}`. The `/ai` HTTP response uses the dispatched call's status code (see §6.4). If the request had `?raw=true`, the dispatched response body and status are returned directly with no envelope.
8. `on_decision` hook is fired with the `Decision` object. For any `AIRouterError` raised during steps 3-7, `on_error` is fired with an `ErrorEvent` before the HTTP response is returned.

### 5.3 OpenAPI → tool schema (technical core)

For each route surviving the mode filter, we project an OpenAPI operation into a `RouteSpec`:

```python
@dataclass(frozen=True)
class RouteSpec:
    name: str                          # tool name the LLM sees
    description: str                   # from @ai_route(description=...) or docstring
    method: str                        # "POST"
    path_template: str                 # "/orders/{order_id}/cancel"
    parameters_schema: dict            # merged JSON Schema, flat
    param_locations: dict[str, str]    # {"order_id":"path", "reason":"body"}
    handler: Callable                  # for direct-call escape hatch
```

The flat `parameters_schema` is built by merging:

- **Path params** → top-level properties, marked `required`.
- **Query params** → top-level properties, optional unless required by FastAPI.
- **Body fields** (Pydantic model) → flattened into top-level properties.
- **Headers / cookies / `Depends()`** → **excluded.** Auth and request metadata come from the original `/ai` request, not from the LLM.

Tool name resolution priority: `route.name` (FastAPI default) → operation_id → derived from path+method. Conflicts are resolved deterministically by appending the lowercased HTTP method, then a stable hash of the path if needed (e.g., `cancel_order_post`, `cancel_order_post_a3f1`). Conflicts emit a startup warning log so devs can rename to avoid the suffix.

#### 5.3.1 Schema flattening — edge cases

The flat top-level merge applies only to the *boundary* between path/query/body. Internal structure of body fields is preserved as native JSON Schema:

| Body shape | Flattening behavior |
|---|---|
| Pydantic model with scalar fields (`name: str, age: int`) | Each field becomes a top-level property of the parameters_schema. |
| Pydantic model with nested model (`customer: CustomerInfo`) | `customer` is a top-level property of type `object` with the nested schema preserved. We do **not** recursively flatten — tool-calling APIs handle nested objects natively. |
| Multiple `Body(...)` params (`name: str = Body(...), age: int = Body(...)`) | Each becomes a top-level property. FastAPI already merges these into a single body object. |
| Body-as-list (`items: list[Item] = Body(...)`) | Wrapped: parameters_schema gets a single top-level property `items` of type `array`. Dispatcher unwraps before the loopback call. |
| No body (e.g., GET routes) | Body section omitted; only path/query merged. |

**Name collision handling:** If a path param name collides with a body field name (e.g., `/orders/{id}` plus body `Order(id: int, ...)`), introspection logs a startup warning and the body field is renamed in the tool schema with a `_body` suffix. The dispatcher reverses the rename when un-flattening.

**Excluded routes:** Routes whose body content-type is `application/x-www-form-urlencoded` or `multipart/form-data`, or whose request requires file uploads, are excluded from AI exposure with a startup warning. They reappear in v0.2.

### 5.4 Dispatcher (un-flattening + loopback)

```python
from urllib.parse import quote

async def dispatch(
    spec: RouteSpec,
    args: dict,
    original_request: Request,
    forward_headers: frozenset[str],
) -> httpx.Response:
    path_args, query_args, body_args = split_by_location(args, spec.param_locations)

    # URL-encode path params — LLM may return strings containing '/', '?', etc.
    encoded_path_args = {k: quote(str(v), safe="") for k, v in path_args.items()}
    url = spec.path_template.format(**encoded_path_args)

    headers = forward_relevant_headers(original_request, allowed=forward_headers)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=original_request.app),
        base_url="http://airouter.local",
    ) as client:
        return await client.request(
            method=spec.method,
            url=url,
            params=query_args or None,
            json=body_args if body_args else None,
            headers=headers,
        )
```

Two implementation details that matter:

- **URL-encoding of path arguments is mandatory.** The LLM may return string values containing `/`, `?`, `#`, etc. Without `urllib.parse.quote`, these mangle the URL and cause silent route mismatches. Tests must cover this.
- **The httpx client is created per-request for ASGITransport.** ASGITransport doesn't pool real connections (no network), so per-request is fine. We do *not* hold a long-lived client on the AIRouter instance — that would make testing the dispatcher harder and offer no real performance benefit.

This pattern is the same one FastAPI's `TestClient` uses, so it is a battle-tested approach to in-process request loopback.

### 5.5 Caching policy

The cache lives on the `AIRouter` instance, keyed implicitly by the (`app`, `mode_config`) the instance was constructed with.

- **Build trigger:** built lazily on the first request to `/ai`. Construction-time building is avoided because users may add routes to `app` between `AIRouter(...)` and the first request.
- **Lifetime:** built once, kept for the lifetime of the `AIRouter` instance. Routes added at runtime after the first `/ai` request are NOT picked up automatically.
- **Manual rebuild:** `AIRouter.rebuild()` clears the cache; the next request rebuilds. Documented as the official escape hatch for runtime route changes (rare, but real for some test setups).
- **Hot-reload during development:** supported transparently — uvicorn `--reload` restarts the worker, which reinstantiates the `AIRouter`, which rebuilds on first request.

We deliberately do **not** watch `app.routes` for changes (no observers, no signals). The simplicity of "build once, optional manual rebuild" is more valuable than auto-detection of an uncommon case.

---

## 6. Auth, errors, observability

### 6.1 Two-layer auth

Auth is enforced in two layers, both via FastAPI's existing mechanisms — the library does not implement any authentication itself.

- **Layer 1:** `dependencies=[Depends(...)]` passed to `AIRouter(...)` is applied to the `/ai` endpoint. Gates "who can use the AI feature at all."
- **Layer 2:** The dispatched route's own `Depends()` chain runs unchanged during the loopback call. Gates "who can call this specific endpoint."

Both must pass. If Layer 2 rejects the dispatched call (e.g., 401/403), the original client receives the envelope with `result_status` reflecting the rejection.

### 6.2 Header forwarding

Default forwarded headers (auth + tracing only, conservative):

```python
DEFAULT_FORWARD_HEADERS = {
    "authorization",
    "cookie",
    "x-api-key",
    "x-forwarded-for",
    "x-request-id",
}
```

Devs can override via `forward_headers=` to add tenant headers, etc.

### 6.3 Error taxonomy

```python
class AIRouterError(Exception):
    """Base — all library errors derive from this."""

class NoRouteMatched(AIRouterError):
    """LLM declined to call any tool. Carries the LLM's text reply."""

class UnknownTool(AIRouterError):
    """LLM called a tool name we didn't expose."""

class LLMBackendError(AIRouterError):
    """LLM call itself failed (timeout, rate limit, invalid response).
    Wraps the upstream exception."""

class DispatchError(AIRouterError):
    """Loopback HTTP call failed for non-route-logic reasons (transport error)."""

class ToolSchemaTooLarge(AIRouterError):
    """Combined tool definitions exceed the configured token budget.
    Recommend mode='decorator' filtering or upgrade to v0.2 prefiltering."""
```

### 6.4 HTTP surface for errors

| Failure | HTTP status | Envelope shape |
|---|---|---|
| `NoRouteMatched` | 422 | `{"error":"no_route_matched", "reasoning":"<llm text>", "available_tools":[...]}` |
| `UnknownTool` | 422 | `{"error":"unknown_tool", "tool_name":"...", "available_tools":[...]}` |
| `LLMBackendError` | 502 | `{"error":"llm_backend_error", "detail":"...", "retryable":true/false}` |
| Dispatched route 4xx/5xx | the `/ai` response uses the **same status code** as the dispatched route | envelope wraps it: `{"endpoint":"...", "result":{...}, "result_status":403}`. Rationale: clients' normal HTTP error handling continues to work; the library does not mask downstream errors as 200 OK. |
| `DispatchError` | 500 | `{"error":"dispatch_error", "detail":"..."}` |
| `ToolSchemaTooLarge` | 500 | `{"error":"tool_schema_too_large", "tool_count":N, "approx_tokens":T}` |

Dispatched-route errors are not swallowed. Route-level authorization is the source of truth for whether a call is allowed.

### 6.5 Observability hooks

Two async hooks. No vendor dependencies in the core. Both are awaited by default; users can wrap them in `asyncio.create_task` for fire-and-forget if they want to absorb latency.

```python
@dataclass(frozen=True)
class Decision:
    request_id: str
    query: str
    tool_name: str | None        # None when no tool was selected
    args: dict
    reasoning: str | None
    model: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: int
    dispatch_latency_ms: int | None
    result_status: int | None    # HTTP status of the dispatched call

@dataclass(frozen=True)
class ErrorEvent:
    request_id: str
    query: str
    error_type: str
    error_detail: str
    upstream: BaseException | None

class DecisionHook(Protocol):
    async def __call__(self, decision: Decision) -> None: ...

class ErrorHook(Protocol):
    async def __call__(self, event: ErrorEvent) -> None: ...
```

Plus `logging.getLogger("fastapi_ai_router")`:

- INFO: each Decision (without prompt content)
- ERROR: each ErrorEvent
- DEBUG: full prompt, response, tool definitions (only when `debug=True`)

---

## 7. LLM backend abstraction

### 7.1 Protocol

```python
@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict                       # JSON-decodable; matches the tool's parameters_schema
    reasoning: str | None            # backend-supplied "thought" if available
    prompt_tokens: int               # 0 if backend cannot report
    completion_tokens: int           # 0 if backend cannot report
    model: str                       # the model identifier the backend used

class LLMBackend(Protocol):
    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None: ...
```

**Wire shapes** (the core constructs these and passes them in; backends translate to whatever their vendor SDK expects):

```python
class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class ToolDef(TypedDict):
    type: Literal["function"]
    function: FunctionDef

class FunctionDef(TypedDict):
    name: str
    description: str
    parameters: dict                 # JSON Schema object
```

This is the OpenAI-compatible function-calling shape, which is also what Anthropic, Gemini, and most OSS function-calling models support natively. A backend's job is to translate this in→out:

- in: `(messages, tools)` in the wire shape above
- out: `ToolCall` (the chosen tool + args) or `None` if the model declined to call any tool

The core depends only on this protocol. No vendor SDKs in `core.py`.

### 7.2 Shipped backends

- **`LiteLLMBackend(model=...)`** — convenience adapter that wraps LiteLLM. Imported from `fastapi_ai_router.backends.litellm`. Requires `pip install fastapi-ai-router[litellm]`.
- **`FakeLLMBackend(returns=...)`** — for tests. Returns a pre-canned `ToolCall`. Zero dependencies.

Community-contributed backends (Anthropic-direct, Ollama, vLLM, etc.) are encouraged via `CONTRIBUTING.md` but not part of v0.1's core.

---

## 8. Testing strategy

Three tiers:

1. **Unit tests** — fast, no I/O, no API keys. `FakeLLMBackend` everywhere. ~80% of total tests.
2. **Integration tests** — FastAPI's `TestClient`, full loopback, still `FakeLLMBackend`. Verifies HTTP plumbing.
3. **Real-LLM smoke tests** — gated behind `RUN_LLM_TESTS=1`. Three or four tests against a real model, run on release tags only.

Coverage target: **80%+** enforced via `pytest --cov=fastapi_ai_router --cov-fail-under=80`.

Behaviors that must have coverage:

- Each `mode` exposes the right routes (and only those).
- `include_in_schema=False` routes are never exposed regardless of mode.
- Path/query/body un-flattening for varied param combinations.
- Pydantic body model fields flatten correctly into the tool schema.
- `Authorization` header forwarding to dispatched route.
- Layer-2 auth fires (`Depends(auth)` on the dispatched route still rejects unauthorized calls).
- Each error type maps to the correct HTTP status and envelope shape.
- `on_decision` and `on_error` hooks fire with correct payloads.
- `?raw=true` bypasses envelope.
- Decorator-mode skips routes without `@ai_route`.

---

## 9. Repo structure

```
fastapi-ai-router/
├── src/fastapi_ai_router/        # package (modules per §5.1)
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/                      # gated behind RUN_LLM_TESTS
├── examples/
│   ├── 01_basic.py
│   ├── 02_tag_mode.py
│   ├── 03_with_auth.py
│   └── 04_with_observability.py
├── docs/
│   ├── concepts.md
│   ├── recipes.md
│   ├── security.md
│   └── api.md
├── .github/workflows/
│   ├── test.yml                  # ruff + mypy + pytest on 3.11/3.12/3.13
│   └── publish.yml               # PyPI release on tag push
├── pyproject.toml                # uv-managed, ruff + mypy strict
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE                       # MIT
```

---

## 10. Milestones (timebox-honest)

### v0.1.0 — target 6 weeks part-time, scope-locked

- Core: introspection, dispatch, envelope, errors, observability hooks.
- 3 modes: `decorator` / `tag` / `all`.
- 2 backends: `LiteLLMBackend`, `FakeLLMBackend`.
- 80%+ test coverage; deterministic test suite without API keys.
- README, concepts, security, recipes docs.
- Published to PyPI as `fastapi-ai-router`.

### v0.2.0 — post-launch, demand-driven (priority order)

1. Streaming response (SSE).
2. Mountable sub-app pattern.
3. Semantic prefiltering for apps with many routes.
4. Community-contributed backends (Anthropic direct, Ollama).

### v0.3.0+ — only with adoption pull

- Multi-step / `max_steps` agent loop.
- Conversation state hooks.
- Heuristic content guard against prompt-injection patterns.

---

## 11. Documentation plan

| Doc | Purpose |
|---|---|
| `README.md` | Hook in 30 seconds → 5-line demo → install + first call in under 2 minutes. Includes a **Limitations** section listing the v0.1 non-goals from §3. |
| `docs/concepts.md` | Mental model: how routes become tools, request flow, two-layer auth, mode comparison. |
| `docs/recipes.md` | Cookbook: custom backend, custom forwarding, observability integrations (Langfuse / OTel / Sentry / plain logs), large-app strategies. |
| `docs/security.md` | When `mode="all"` is dangerous, prompt-injection considerations, header-forwarding implications, recommended practices. |
| `docs/api.md` | Auto-generated reference from docstrings. |
| `CONTRIBUTING.md` | How to run tests without API keys, code style, PR conventions, how to add a new backend. |

---

## 12. Open questions (resolved during implementation)

1. **Default system prompt** for the LLM. Needs prompt-engineering iteration; benchmark against real apps before locking.
2. **Token budget management.** v0.1 errors with `ToolSchemaTooLarge` and recommends `mode="decorator"` filtering. v0.2 adds semantic prefiltering. The exact threshold for "too large" must be configurable and conservatively defaulted.
3. **Form bodies / file uploads.** `application/x-www-form-urlencoded` and multipart are out of scope for v0.1; routes that require them should not be exposed (introspection should detect and warn at startup).
4. **Prompt injection.** Mitigations: constrained tool surface, defense-in-depth via Layer-2 auth, explicit warnings in `docs/security.md`. v0.2 may add a heuristic content guard.
5. **Async vs sync route handlers.** Both are supported by FastAPI. The dispatcher uses `httpx.AsyncClient` regardless; sync handlers run in FastAPI's threadpool fallback. Verify in integration tests.
6. **Pydantic v1 vs v2.** Assume v2 only; document.

---

## 13. Decision log

For traceability, the major design choices made during brainstorming:

| Decision | Choice | Why |
|---|---|---|
| Product type | OSS library, not SaaS | Maximize adoption; positioned as the FastAPI conversational layer. |
| Scope | NL → endpoint dispatch only | Sharp wedge; avoids competing with LangChain on agents. |
| Activation modes | All three (`decorator` / `tag` / `all`), explicit choice, no silent fallback | Flexibility without footguns. |
| Default mode | `decorator` | Safe by default. |
| Request shape | Dedicated endpoint, default `/ai`, configurable | Cleanest mental model; demoable in one curl line. |
| Response shape | Wrapped envelope, `?raw=true` for pass-through | Critical for debugging LLM decisions. |
| LLM abstraction | BYO via Protocol; LiteLLM-backed default | Decouples core from vendor churn; testable. |
| Single-shot vs multi-step | Single-shot only in v0.1 | Stays out of agent-framework territory. |
| Tool schemas | Extracted from FastAPI's OpenAPI, flat parameter merging | Native LLM tool-calling format; no new metadata required. |
| Dispatch mechanism | httpx + ASGITransport loopback | Auth, validation, middleware all "just work." |
| Auth model | Two-layer: Layer-1 on `/ai`, Layer-2 on dispatched route via header forwarding | Re-uses FastAPI's primitives; no auth reimplementation. |
| Observability | Async hooks, no vendor deps | Lets every team plug in their stack. |
