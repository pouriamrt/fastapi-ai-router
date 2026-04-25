# Concepts

## Mental model

`AIRouter(app, llm=...)` adds a single `POST /ai` endpoint. That endpoint:

1. Builds a list of LLM tool definitions from your existing FastAPI routes
   (filtered by `mode`).
2. Asks `llm` to pick exactly one tool given the user's natural-language query.
3. Calls that tool's underlying route via an in-process HTTP loopback.
4. Returns either an envelope or the raw response (if `?raw=true`).

Nothing about your existing routes changes. They keep their auth, middleware,
validation, and exception handlers. AIRouter is a *consumer* of routes, not a
modifier.

## Request flow

```
client → POST /ai → AIRouter
                      ↓ Layer-1 deps (e.g. paid-tier check)
                      ↓ build tools from app.routes
                      ↓ llm.call(messages, tools)
                      ↓ resolve tool → RouteSpec
                      ↓ dispatcher: split args, URL-encode path, loopback
                                              ↓
                                       FastAPI runs route's full pipeline:
                                         middleware → Depends → validation
                                         → handler → exception handler
                      ← envelope (or raw) ←
client ← POST /ai response (status mirrors dispatched call)
```

## Two-layer auth

- **Layer 1:** `dependencies=` you pass to `AIRouter()` is applied to `/ai`
  itself. Use it to gate the AI feature.
- **Layer 2:** Each dispatched route's own `Depends(...)` runs unchanged via
  the loopback call. The original request's `Authorization` (and other
  configured) headers are forwarded.

Both must pass. Layer-1 failures never make an LLM call. Layer-2 failures pass
through with the dispatched status code (e.g., 401, 403).

## Mode comparison

See the [Exposure modes table](../README.md#exposure-modes) in the README.

`expose=False` on `@ai_route` is a kill switch — it excludes the route from
exposure in *every* mode. Use it for routes that exist for humans but should
never be reachable by NL.

## Caching

The registry is built lazily on the first `/ai` request and cached on the
`AIRouter` instance. Routes added at runtime after the first request are NOT
picked up; call `router.rebuild()` to invalidate the cache.

## What's not in v0.1

See the [Limitations](../README.md#limitations-v01) section. v0.2 candidates
in priority order: streaming SSE, mountable sub-app, semantic prefiltering,
community backends.
