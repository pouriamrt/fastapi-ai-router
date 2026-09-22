# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `JevBackend` hands a request to `fallback` when Jev rejects it as invalid (HTTP 400), for example an app with 255 or more routes or a request over Jev's context limit. Without a fallback, the error still surfaces as `llm_backend_error`.
- The `DecisionHook` and `ErrorHook` parameters are positional-only, so type checkers accept hooks that name the parameter anything, such as `async def hook(d)`.

### Fixed
- `$defs` from query params, a lone non-model body, and multiple `Body()` params now sit at the root of the tool schema, where their `#/$defs/...` refs resolve. Only flattened body models did this before. When two params use the same def name for different schemas, the later one is renamed (for example `Mode__second`) rather than overwriting the other.

## [0.2.1] — 2026-09-21

### Changed
- Releases publish from GitHub Actions through PyPI trusted publishing, in the `pypi` environment.

### Fixed
- The source distribution contains only the package, tests, examples, and docs. The 0.2.0 sdist also shipped local tool caches, including a 3.4 MB code-graph database with absolute file paths, plus internal planning docs. The 0.2.0 wheel was not affected.

## [0.2.0] — 2026-09-21

Removed from PyPI; install 0.2.1, which has the same code with a clean sdist.

### Added
- `JevBackend` (`fastapi-ai-router[jev]`): routes with TypeSafe's Jev classifier in a single request, with no LLM call. An optional `fallback=` backend handles low route confidence and arguments Jev can't extract.
- `confidence` field on `ToolCall` and `Decision`, defaulting to `None`.
- `MissingPathParams` error.
- A `py.typed` marker, so type checkers use the package's inline type hints (PEP 561).

### Changed
- Dependencies refreshed. Dev-tool floors raised to ruff 0.16, mypy 2.3, pytest 9.1.

### Fixed
- A backend that leaves out a path argument now gets `422 missing_path_param` instead of `500 dispatch_error`.
- Enum fields on flattened body models keep their `$defs`, so their `$ref`s resolve.
- A route whose only body param isn't a plain model (a list, dict, scalar, or `Model | None`) now receives the bare value FastAPI expects, and a `Body(embed=True)` model is sent keyed by name. These shapes used to fail with 422, and a lone `dict` body silently received `{name: value}` instead of the value.

## [0.1.0] — 2026-04-25

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
