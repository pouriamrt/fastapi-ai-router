# Contributing

Thanks for your interest in fastapi-ai-router. This document explains how to
develop and submit changes.

## Local development

```bash
git clone https://github.com/pouria/fastapi-ai-router
cd fastapi-ai-router
uv sync --extra dev
uv run pytest                    # full suite (no API keys needed)
uv run ruff check .
uv run mypy
```

The test suite is **deterministic and network-free** by default — every test
uses `FakeLLMBackend`. You should never need an API key to contribute.

## Testing against a real LLM

The e2e suite is gated:

```bash
RUN_LLM_TESTS=1 OPENAI_API_KEY=... uv run pytest tests/e2e/ -v
```

Runs only on release tags in CI; PRs do not run e2e.

## Adding a new LLM backend

1. Create `src/fastapi_ai_router/backends/<vendor>.py`.
2. Subclass nothing — implement the `LLMBackend` Protocol structurally.
3. Translate `(messages, tools)` to the vendor's function-calling format.
4. Wrap any vendor exceptions in `LLMBackendError(... upstream=exc)`.
5. Add unit tests using `unittest.mock.patch` on the vendor's async client.
6. Add an example in `examples/` demonstrating the new backend.

## Code style

- `ruff` for lint and format (`uv run ruff check . --fix && uv run ruff format .`)
- `mypy --strict` for type checking
- Test names: `test_<thing>_<does>` (snake_case, descriptive)
- Commits: conventional commits (`feat(scope): ...`, `fix(scope): ...`, `docs: ...`)

## PR conventions

- One feature per PR.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Add tests for new behavior; keep coverage at 80%+.
