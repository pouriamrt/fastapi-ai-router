# JevBackend + dependency refresh — design

Date: 2026-09-21
Status: draft, awaiting review

## Goal

Let `AIRouter` route natural-language requests with TypeSafe's Jev model, and refresh the project's dependencies.

Jev is the reason for doing this at all: it is fast and cheap. The design keeps a request on Jev alone whenever it can, and calls an LLM only when the user has opted in to a fallback and Jev cannot finish the job.

## What Jev is (and is not)

Jev is a "System One" classifier, not a chat model. A request is a `state` (text) plus named typed questions: `Choice` (pick one option), `Score` (ordered levels), `Noul` (yes/no probability). Each answer carries calibrated probabilities and a `confidence`. It has no text generation, no function calling, and no OpenAI-compatible endpoint, so `LiteLLMBackend` cannot reach it.

Facts checked against the live API on 2026-09-21 (`jev-1.13.0`, SDK `typesafe-sdk` 0.7.1):

- `AsyncTypeSafeClient(api_key=None, model="jev-latest", ...)`; the key defaults to the `TYPESAFE_API_KEY` env var.
- A `Choice` accepts at most 255 options. 256 returns `400 Too many choices`.
- A one-option `Choice` is accepted. Option keys may contain punctuation and quotes.
- Price is $0.042 per million input tokens; output tokens are free.

## The approach: span selection with speculative fan-out

Jev cannot write `order_id=123`, but it can pick `"123"` from a list. So code builds candidate values out of the query text and Jev selects among them. The route question and a question for every parameter of every route go out in a single request ("speculative fan-out" in TypeSafe's docs); code then reads only the answers for the route Jev picked.

A throwaway spike ran seven queries against two example routes. All seven routes and all argument values came back correct, including `not_stated` for omitted optional params and `none_of_the_above` for an off-topic query. Latency was 120–350 ms per request at 800–1,500 input tokens (about $0.00006 each).

## Components

### `backends/jev.py` (new)

```python
@dataclass
class JevBackend(LLMBackend):
    model: str = "jev-latest"
    api_key: str | None = None          # None: SDK reads TYPESAFE_API_KEY
    min_confidence: float = 0.5         # route-confidence gate
    fallback: LLMBackend | None = None  # None: never calls an LLM
    max_span_words: int = 6             # longest n-gram offered for string params
    client: AsyncTypeSafeClient | None = None  # injectable; created lazily otherwise
```

Imported lazily behind a new extra, `jev = ["typesafe-sdk>=0.7.1"]`, the same way `litellm.py` guards its import. The SDK ships `py.typed`, so mypy strict needs no ignores for it.

The module holds small pure functions (param classification, candidate generation, question building, answer coercion) plus the `call()` method that ties them together. The pure functions are what the unit tests exercise.

### Public types (additive change)

`ToolCall` and `Decision` each gain `confidence: float | None = None`, and `core.py` copies it from one to the other. The default keeps `LiteLLMBackend`, `FakeLLMBackend`, and user-written backends working unchanged.

## Data flow for one request

1. **Inputs.** The query is the content of the last `user` message; system messages are ignored. Route names, descriptions, and parameter schemas come from the `ToolDef`s that `core.py` already passes in. Core does not change here.

2. **Classify each parameter** from its JSON schema:
   - Unwrap `anyOf: [X, {"type": "null"}]` to `X`.
   - Resolve `{"$ref": "#/$defs/Name"}` against the property's own `$defs`.
   - Supported: `integer`, `number`, `boolean`, `string`, and `enum`. Anything else (`object`, `array`, an unresolvable `$ref`) is unsupported.

3. **Build candidates** from the query:
   - `integer`: every integer token, e.g. `#4521,` gives `4521`.
   - `number`: every integer or decimal token.
   - `string`: every contiguous word n-gram up to `max_span_words` long, punctuation stripped, de-duplicated.
   - `enum`: the enum values.
   - `boolean`: `true`, `false`.

   Every question also gets a `not_stated` option. Candidates are capped at 254 so the total stays within Jev's 255-option limit. When a parameter has no candidates at all, the backend skips its question and treats it as `not_stated`.

4. **Send one request:** a `route` Choice over the tool names plus `none_of_the_above`, and one Choice per supported (route, param) pair.

5. **Decide:**

   | Situation | With `fallback=None` | With a fallback |
   |---|---|---|
   | `none_of_the_above`, confidence ≥ `min_confidence` | return `None` (`no_route_matched`) | same; no LLM call |
   | Any route choice, `none_of_the_above` included, with confidence < `min_confidence` | return `None` | fallback with **all** tools |
   | Chosen route has a *required* unsupported param, or a required param came back `not_stated` | return the partial `ToolCall`; the request ends in a 422 naming the missing field (see "Pre-existing bugs fixed alongside") | fallback with **only the chosen route's tool** |
   | Optional param is unsupported or `not_stated` | omit it | omit it |

   Arguments are not gated on confidence. Overlapping string spans ("because it was a duplicate" vs "it was a duplicate") split probability and lower confidence even when the pick is right; the spike saw 0.60 on a correct answer.

6. **Coerce and return.** `int()`/`float()` numbers, map `true`/`false` to bools, pass enum and string values through. A value that fails to coerce is treated as `not_stated`. Return `ToolCall(name, args, reasoning=None, prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens, model=response.model, confidence=route.confidence)`.

   When the fallback runs with the narrowed tool, its `ToolCall` is returned with `confidence` set to Jev's route confidence, because Jev still chose the route. When the fallback runs with all tools, its `ToolCall` is returned as-is.

## Error handling

- Any `TypeSafeError` from the SDK is wrapped in `LLMBackendError`, which core already maps to a 502 with `retryable: true`. The SDK retries 429s with backoff on its own.
- A response missing an expected answer id raises `LLMBackendError`.
- Errors raised by the fallback propagate unchanged; `LiteLLMBackend` already raises `LLMBackendError`.
- The SDK client is created once and reused across requests, which avoids a TLS handshake per call. It is not closed at shutdown, since `AIRouter` has no lifecycle hook. Marked with a `ponytail:` comment.

## Known limits (documented, not solved)

- Number words ("five laptops") are not extracted; only digits are.
- Long queries: string candidates are truncated at the 254 cap.
- Optional unsupported params (lists, nested objects) are dropped silently when Jev handles the request.
- Token cost grows with routes × params × candidates. Jev's context limit is 64k tokens per request. If large apps hit it, a two-call variant (route first, then that route's params) is the upgrade path. Marked with a `ponytail:` comment.
- Jev reads text literally and is not hardened against adversarial input (TypeSafe's own jaggedness notes). The route confidence gate plus FastAPI validation are the backstop.

## Pre-existing bugs fixed alongside

Both affect every backend today, LiteLLM included, and both were reproduced on 2026-09-21. Each gets its own `fix:` commit and regression test.

1. **Dangling `$ref` on body-model enums.** `schema.py` flattens a single body model's `properties` but drops the model's `$defs`, so an enum field ships as `{"$ref": "#/$defs/Color"}` with nothing to resolve it against. The fix copies the needed `$defs` into each flattened property.

2. **Missing path argument returns 500.** `dispatcher.py` fills the URL with `path_template.format(...)`. When the backend omits a path parameter, that raises a bare `KeyError`, and the caller gets `500 {"error": "dispatch_error", "detail": "'order_id'"}`. A missing query or body parameter already gets FastAPI's 422. The fix checks for missing path parameters before formatting, and core returns `422 {"error": "missing_path_param", "missing": ["order_id"], "endpoint": "POST /orders/{order_id}/cancel"}`. It fires `on_error` with `error_type="missing_path_param"`.

## Commit order

1. `chore(deps)`: dependency refresh (below).
2. `fix(schema)`: keep `$defs` for flattened body-model fields.
3. `fix(dispatch)`: 422 instead of 500 for a missing path argument.
4. `feat`: `confidence` field on `ToolCall`/`Decision`, `JevBackend`, the `jev` extra.
5. `docs`: README, example, CHANGELOG.

## Dependency refresh

This goes in its own `chore(deps)` commit, before the Jev work:

- Commit the stale `uv.lock` change (`0.1.0.dev0` to `0.1.0`).
- `uv lock --upgrade` (about 60 packages).
- Raise floors for dev tools only, to the versions now locked: ruff 0.16, mypy 2.3, pytest 9.1, pytest-asyncio 1.4, pytest-cov, python-multipart. Runtime floors (`fastapi>=0.110`, `httpx>=0.27`, `pydantic>=2.0`, `litellm>=1.40`) stay where they are, so library users are not forced to upgrade.
- Fix whatever new findings ruff 0.16 and mypy 2 report, then run the full suite.

`typesafe-sdk` requires `pydantic>=2.12`. That constraint applies only to users who install the `jev` extra.

## Testing

- **Unit** (`tests/unit/test_backends_jev.py`): schema classification (anyOf-null, `$ref`/`$defs`, unsupported types), candidate generation (punctuation, de-duplication, the 254 cap, the empty-candidate skip), question building, coercion, and every row of the decision table. These use a stub client whose `system_one` returns canned answers; no network.
- **Unit:** `confidence` passes from `ToolCall` to `Decision` in core; a regression test for the `$defs` fix.
- **Integration:** `AIRouter` with `JevBackend` and a stub client covers the happy path, no-match, and fallback to `FakeLLMBackend`.
- **E2E** (gated by `RUN_LLM_TESTS=1` plus `TYPESAFE_API_KEY`, like the existing LLM smoke tests): the seven spike queries against the real API. The key lives in the repo's gitignored `.env` and is loaded with `uv run --env-file .env pytest ...`, so no `python-dotenv` dependency is needed.
- Coverage stays at or above the existing 80% gate. `ruff check`, `ruff format --check`, and `mypy` pass.

## Docs

- README: a "Jev backend" section covering install (`fastapi-ai-router[jev]`), the env var, the fallback option, and the known limits.
- `examples/05_jev.py`: the basic example running on `JevBackend`.
- CHANGELOG: an "Unreleased" entry. No version bump or release unless asked.

## Out of scope

- Extracting lists or nested objects with Jev.
- Number-word parsing.
- The two-call variant for very large route sets.
- Exposing per-argument confidences or probabilities.
