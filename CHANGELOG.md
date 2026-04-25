# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release.
- `AIRouter` middleware turning FastAPI routes into an LLM-callable surface.
- Three exposure modes (`decorator`, `tag`, `all`) with safe-by-default decorator mode.
- `@ai_route` decorator with `expose=False` kill switch.
- `LLMBackend` Protocol with `LiteLLMBackend` (default) and `FakeLLMBackend` (tests).
- Two-layer auth via `httpx + ASGITransport` loopback.
- Async `on_decision` and `on_error` observability hooks.
- Wrapped envelope response with `?raw=true` bypass.
- Five exception types (`NoRouteMatched`, `UnknownTool`, `LLMBackendError`, `DispatchError`, `ToolSchemaTooLarge`).
