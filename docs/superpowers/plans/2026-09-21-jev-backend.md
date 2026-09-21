# JevBackend + Dependency Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `JevBackend` that routes `/ai` requests with TypeSafe's Jev classifier in one request and no LLM, fix two routing bugs it exposes, and refresh the project's dependencies.

**Architecture:** `JevBackend` implements the existing `LLMBackend` protocol, so `core.py` barely changes. Jev picks the route from a `Choice` over tool names and picks each argument from a `Choice` over candidate values cut out of the query. All questions go out in one request. An optional `fallback` backend (e.g. `LiteLLMBackend`) handles low route confidence and arguments Jev cannot supply.

**Tech Stack:** Python 3.11+, FastAPI, httpx, pydantic 2, `typesafe-sdk` 0.7.1 (Jev), LiteLLM (optional fallback), pytest + pytest-asyncio (`asyncio_mode = "auto"`), ruff, mypy strict, uv.

**Spec:** `docs/superpowers/specs/2026-09-21-jev-backend-design.md`

## Global Constraints

- Branch: `feat/jev-backend` (already created; the spec is committed there).
- Always `uv run ...`, never bare `python`/`pytest`.
- Runtime floors stay as they are: `fastapi>=0.110`, `httpx>=0.27`, `pydantic>=2.0`, `litellm>=1.40`.
- New extra: `jev = ["typesafe-sdk>=0.7.1"]`; `typesafe-sdk>=0.7.1` is also added to the `dev` extra (CI runs `uv sync --extra dev`).
- API key env var: `TYPESAFE_API_KEY`, stored in the repo's gitignored `.env`. Load it with `uv run --env-file .env ...`. Never print, log, or commit it.
- Jev limits: a `Choice` takes at most 255 options (256 returns `400 Too many choices`). Sentinel option names: `NOT_STATED = "not_stated"`, `NO_ROUTE = "none_of_the_above"`.
- After every task, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest -q` all pass. Coverage stays at or above 80%.
- Conventional commit messages, no attribution trailer. Stage files by name, never `git add -A`.
- Do not modify `examples/01_*.py` through `examples/04_*.py`.

## File Map

| File | Change | Responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock` | modify | dev floors, `jev` extra |
| `src/fastapi_ai_router/schema.py` | modify | hoist body-model `$defs` |
| `src/fastapi_ai_router/errors.py` | modify | `MissingPathParams` |
| `src/fastapi_ai_router/dispatcher.py` | modify | raise `MissingPathParams` before formatting the URL |
| `src/fastapi_ai_router/core.py` | modify | map `MissingPathParams` to 422; pass `confidence` into `Decision` |
| `src/fastapi_ai_router/__init__.py` | modify | export `MissingPathParams` |
| `src/fastapi_ai_router/backends/__init__.py` | modify | `ToolCall.confidence` |
| `src/fastapi_ai_router/observability.py` | modify | `Decision.confidence` |
| `src/fastapi_ai_router/backends/jev.py` | create | `JevBackend` and its pure helpers |
| `tests/conftest.py` | modify | `JevStubClient` |
| `tests/unit/test_backends_jev.py` | create | helper and backend unit tests |
| `tests/integration/test_jev_backend.py` | create | `AIRouter` + `JevBackend` end to end, stubbed |
| `tests/e2e/test_with_real_jev.py` | create | real Jev calls, gated |
| `README.md`, `CHANGELOG.md`, `examples/05_jev.py`, `examples/README.md` | modify/create | docs |

---

### Task 1: Dependency refresh

**Files:**
- Modify: `pyproject.toml` (the `dev` list under `[project.optional-dependencies]`)
- Modify: `uv.lock`
- Modify (formatting only): whatever `ruff format` rewrites

**Interfaces:**
- Consumes: nothing.
- Produces: an upgraded lockfile and toolchain (ruff 0.16.x, mypy 2.3.x, pytest 9.1.x) that later tasks run under.

Pre-checked on 2026-09-21: ruff 0.16.8 `check` and mypy 2.3.1 both pass on the current code with no new findings. ruff 0.16 `format` would rewrite 15 files, `README.md` code blocks among them. That drift already exists and gets its own `style` commit.

- [ ] **Step 1: Upgrade the lockfile**

Run: `uv lock --upgrade && uv sync --extra dev`
Expected: about 60 packages updated, including `fastapi v0.141.x`, `litellm v1.102.x`, `mypy v2.3.x`, `ruff v0.16.x`, `pytest v9.1.x`.

- [ ] **Step 2: Raise the dev-tool floors**

In `pyproject.toml`, replace the `dev` list with:

```toml
dev = [
    "pytest>=9.1",
    "pytest-asyncio>=1.4",
    "pytest-cov>=7.1",
    "ruff>=0.16",
    "mypy>=2.3",
    "litellm>=1.40",
    "python-multipart>=0.0.32",
]
```

Leave `dependencies` and the `litellm` extra untouched.

- [ ] **Step 3: Re-lock and sync**

Run: `uv lock && uv sync --extra dev`
Expected: resolves without errors.

- [ ] **Step 4: Verify the toolchain on the unchanged code**

Run: `uv run ruff check . && uv run mypy && uv run pytest -q`
Expected: `All checks passed!`, `Success: no issues found in 12 source files`, `74 passed, 2 skipped`.

If anything fails, fix it at its source and paste the failing output into your report. No blanket `# type: ignore` or `noqa`.

- [ ] **Step 5: Commit the dependency refresh**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): upgrade lockfile and raise dev-tool floors"
```

- [ ] **Step 6: Apply ruff 0.16 formatting**

Run: `uv run ruff format . && uv run ruff format --check . && uv run pytest -q`
Expected: `N files reformatted`, then `... files already formatted`, then `74 passed, 2 skipped`.

- [ ] **Step 7: Commit the formatting**

```bash
git status --short   # every modified file should be one ruff just reformatted
git add $(git diff --name-only)
git commit -m "style: apply ruff 0.16 formatting"
```

---

### Task 2: Hoist body-model `$defs` to the parameters root

**Files:**
- Modify: `src/fastapi_ai_router/schema.py` (`_build_parameters_schema`)
- Test: `tests/unit/test_schema_flatten.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `RouteSpec.parameters_schema` gains a top-level `"$defs"` key whenever a flattened body model has `$defs`. Task 5 resolves `#/$defs/...` refs against it.

- [ ] **Step 1: Write the failing tests**

Add `from enum import Enum` to the imports at the top of `tests/unit/test_schema_flatten.py`, then append:

```python
def test_body_model_enum_defs_hoisted_to_root():
    class Color(str, Enum):
        red = "red"
        blue = "blue"

    class Paint(BaseModel):
        color: Color

    app = FastAPI()

    @app.post("/paint", name="paint")
    def paint(p: Paint) -> dict:
        return {"ok": True}

    spec = route_to_spec(_route_named(app, "paint"))
    assert spec is not None
    schema = spec.parameters_schema
    assert schema["properties"]["color"] == {"$ref": "#/$defs/Color"}
    assert schema["$defs"]["Color"]["enum"] == ["red", "blue"]


def test_schema_without_defs_has_no_defs_key():
    app = FastAPI()

    @app.get("/products", name="list_products")
    def list_products(limit: int = 20) -> dict:
        return {"items": []}

    spec = route_to_spec(_route_named(app, "list_products"))
    assert spec is not None
    assert "$defs" not in spec.parameters_schema
```

- [ ] **Step 2: Run the tests to verify the first fails**

Run: `uv run pytest tests/unit/test_schema_flatten.py -v`
Expected: `test_body_model_enum_defs_hoisted_to_root` FAILS with `KeyError: '$defs'`; `test_schema_without_defs_has_no_defs_key` passes.

- [ ] **Step 3: Implement**

In `src/fastapi_ai_router/schema.py`, inside `_build_parameters_schema`:

After `locations: dict[str, ParamLocation] = {}` add:

```python
    defs: dict[str, Any] = {}
```

In the single-body-model branch, directly after `schema = annotation.model_json_schema()` add:

```python
            # Field schemas point at "#/$defs/..."; hoist the defs so those refs resolve.
            defs.update(schema.get("$defs") or {})
```

Replace the end of the function:

```python
    schema_obj: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return schema_obj, locations
```

with:

```python
    schema_obj: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if defs:
        schema_obj["$defs"] = defs
    return schema_obj, locations
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q && uv run mypy`
Expected: `76 passed, 2 skipped`; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/fastapi_ai_router/schema.py tests/unit/test_schema_flatten.py
git commit -m "fix(schema): hoist body-model \$defs so enum refs resolve"
```

---

### Task 3: Missing path argument returns 422, not 500

**Files:**
- Modify: `src/fastapi_ai_router/errors.py`
- Modify: `src/fastapi_ai_router/dispatcher.py` (`dispatch`)
- Modify: `src/fastapi_ai_router/core.py` (dispatch `try` block, imports)
- Modify: `src/fastapi_ai_router/__init__.py`
- Test: `tests/unit/test_dispatcher.py`, `tests/integration/test_error_responses.py`, `tests/unit/test_public_api.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `class MissingPathParams(AIRouterError)` with attribute `missing: tuple[str, ...]`, raised by `dispatch()`. `/ai` responds `422 {"error": "missing_path_param", "missing": [...], "endpoint": "<METHOD> <path>"}` and fires `on_error` with `error_type="missing_path_param"`. Task 7 relies on this response.

- [ ] **Step 1: Write the failing unit test**

In `tests/unit/test_dispatcher.py` add `from fastapi_ai_router.errors import MissingPathParams` to the imports, then append:

```python
@pytest.mark.asyncio
async def test_dispatch_raises_missing_path_params():
    spec = RouteSpec(
        name="cancel",
        description="",
        method="POST",
        path_template="/orders/{order_id}/cancel",
        parameters_schema={"type": "object"},
        param_locations={"order_id": "path", "reason": "query"},
        handler=lambda: None,
    )
    with pytest.raises(MissingPathParams) as ei:
        await dispatch(
            spec=spec,
            args={"reason": "dup"},
            app=_build_app(),
            request_headers={},
            forward=frozenset(),
        )
    assert ei.value.missing == ("order_id",)
```

- [ ] **Step 2: Write the failing integration test**

Append to `tests/integration/test_error_responses.py`:

```python
def test_missing_path_param_returns_422(sample_app):
    events = []

    async def on_error(event) -> None:
        events.append(event)

    backend = make_backend(
        ToolCall(
            name="cancel",
            args={"reason": "dup"},
            reasoning=None,
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator", on_error=on_error)
    resp = TestClient(sample_app).post("/ai", json={"query": "cancel my order"})
    assert resp.status_code == 422
    assert resp.json() == {
        "error": "missing_path_param",
        "missing": ["order_id"],
        "endpoint": "POST /orders/{order_id}/cancel",
    }
    assert events[0].error_type == "missing_path_param"
```

In `tests/unit/test_public_api.py`, add `MissingPathParams,  # noqa: F401` to the import list, in alphabetical position after `Message`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_dispatcher.py tests/integration/test_error_responses.py tests/unit/test_public_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'MissingPathParams'`.

- [ ] **Step 4: Add the error**

Append to `src/fastapi_ai_router/errors.py`:

```python
class MissingPathParams(AIRouterError):
    """Backend left out one or more path parameters, so the URL can't be built."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(f"Missing path parameters: {', '.join(missing)}")
```

- [ ] **Step 5: Raise it in the dispatcher**

In `src/fastapi_ai_router/dispatcher.py` add `from fastapi_ai_router.errors import MissingPathParams` after the `RouteSpec` import. In `dispatch`, replace:

```python
    path_args, query_args, body_args = split_by_location(args, spec.param_locations)

    encoded_path_args = {k: quote(str(v), safe="") for k, v in path_args.items()}
```

with:

```python
    path_args, query_args, body_args = split_by_location(args, spec.param_locations)
    missing = tuple(
        name
        for name, loc in spec.param_locations.items()
        if loc == "path" and name not in path_args
    )
    if missing:
        raise MissingPathParams(missing)

    encoded_path_args = {k: quote(str(v), safe="") for k, v in path_args.items()}
```

- [ ] **Step 6: Map it to 422 in core**

In `src/fastapi_ai_router/core.py` change `from fastapi_ai_router.errors import LLMBackendError` to:

```python
from fastapi_ai_router.errors import LLMBackendError, MissingPathParams
```

In `_handle`, in the dispatch `try`, insert this `except` clause **before** the existing `except BaseException as exc:` clause:

```python
        except MissingPathParams as exc:
            await self._fire_error(
                request_id, query, "missing_path_param", str(exc), exc
            )
            return _json_response(
                422,
                {
                    "error": "missing_path_param",
                    "missing": list(exc.missing),
                    "endpoint": f"{spec.method} {spec.path_template}",
                },
            )
```

- [ ] **Step 7: Export it**

In `src/fastapi_ai_router/__init__.py` add `MissingPathParams,` to the `from fastapi_ai_router.errors import (...)` block (after `LLMBackendError`), and add `"MissingPathParams",` to `__all__` (after `"Message"`).

- [ ] **Step 8: Run everything**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: all clean; `78 passed, 2 skipped`.

- [ ] **Step 9: Commit**

```bash
git add src/fastapi_ai_router/errors.py src/fastapi_ai_router/dispatcher.py src/fastapi_ai_router/core.py src/fastapi_ai_router/__init__.py tests/unit/test_dispatcher.py tests/integration/test_error_responses.py tests/unit/test_public_api.py
git commit -m "fix(dispatch): return 422 missing_path_param instead of 500"
```

---

### Task 4: `confidence` on `ToolCall` and `Decision`

**Files:**
- Modify: `src/fastapi_ai_router/backends/__init__.py` (`ToolCall`)
- Modify: `src/fastapi_ai_router/observability.py` (`Decision`)
- Modify: `src/fastapi_ai_router/core.py` (`_fire_decision`)
- Test: `tests/integration/test_observability.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ToolCall.confidence: float | None = None` (last field) and `Decision.confidence: float | None = None` (last field). Core copies one into the other. Task 6 sets it; Task 7 asserts it.

- [ ] **Step 1: Write the failing tests**

In `tests/integration/test_observability.py`, add `assert d.confidence is None` as the last line of `test_on_decision_fires_with_full_payload`, then append:

```python
def test_on_decision_carries_backend_confidence(sample_app):
    captured: list[Decision] = []

    async def hook(d: Decision) -> None:
        captured.append(d)

    backend = make_backend(
        ToolCall(
            name="list_products",
            args={},
            reasoning=None,
            prompt_tokens=0,
            completion_tokens=0,
            model="fake",
            confidence=0.93,
        )
    )
    AIRouter(sample_app, llm=backend, mode="decorator", on_decision=hook)
    TestClient(sample_app).post("/ai", json={"query": "list products"})
    assert captured[0].confidence == 0.93
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/test_observability.py -v`
Expected: FAIL with `TypeError: ToolCall.__init__() got an unexpected keyword argument 'confidence'` and `AttributeError: 'Decision' object has no attribute 'confidence'`.

- [ ] **Step 3: Add the fields**

In `src/fastapi_ai_router/backends/__init__.py`, add a last field to `ToolCall`:

```python
    model: str
    confidence: float | None = None  # calibrated route confidence, if the backend has one
```

In `src/fastapi_ai_router/observability.py`, add a last field to `Decision`:

```python
    result_status: int | None
    confidence: float | None = None
```

In `src/fastapi_ai_router/core.py`, in `_fire_decision`, add to the `Decision(...)` call after `result_status=result_status,`:

```python
            confidence=tool_call.confidence,
```

- [ ] **Step 4: Run everything**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: all clean; `79 passed, 2 skipped`.

- [ ] **Step 5: Commit**

```bash
git add src/fastapi_ai_router/backends/__init__.py src/fastapi_ai_router/observability.py src/fastapi_ai_router/core.py tests/integration/test_observability.py
git commit -m "feat: expose backend confidence on ToolCall and Decision"
```

---

### Task 5: Jev helpers: schema classification, candidates, questions, coercion

**Files:**
- Modify: `pyproject.toml` (optional-dependencies), `uv.lock`
- Create: `src/fastapi_ai_router/backends/jev.py`
- Test: `tests/unit/test_backends_jev.py`

**Interfaces:**
- Consumes: `ToolDef` from `fastapi_ai_router.backends`; the top-level `"$defs"` from Task 2.
- Produces (all in `fastapi_ai_router.backends.jev`):
  - constants `NOT_STATED = "not_stated"`, `NO_ROUTE = "none_of_the_above"`, `ROUTE_QID = "route"`, `MAX_CHOICE_OPTIONS = 255`
  - `ParamKind = Literal["integer", "number", "boolean", "string", "enum"]`
  - `@dataclass(frozen=True) class ArgSlot(qid: str, route: str, param: str, kind: ParamKind | None, required: bool, options: tuple[str, ...], enum_values: tuple[Any, ...] = ())`
  - `classify(schema: Mapping[str, Any], root_defs: Mapping[str, Any]) -> tuple[ParamKind | None, tuple[Any, ...]]`
  - `candidates(query: str, kind: ParamKind, enum_values: tuple[Any, ...], max_span_words: int) -> tuple[str, ...]`
  - `build_slots(query: str, tools: list[ToolDef], max_span_words: int) -> list[ArgSlot]`; slot `qid`s are `"arg0"`, `"arg1"`, ... in tool order, then property order
  - `build_questions(tools: list[ToolDef], slots: list[ArgSlot]) -> dict[str, Choice]`
  - `coerce(slot: ArgSlot, raw: str) -> Any`, which returns `None` when the value doesn't convert

- [ ] **Step 1: Add the extra and install**

In `pyproject.toml` under `[project.optional-dependencies]`, add after the `litellm` line:

```toml
jev = ["typesafe-sdk>=0.7.1"]
```

and add `"typesafe-sdk>=0.7.1",` as the last entry of the `dev` list.

Run: `uv lock && uv sync --extra dev`
Expected: `typesafe-sdk==0.7.1` (plus `httpx2`, `tenacity`) installed.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_backends_jev.py`:

```python
import pytest

from fastapi_ai_router.backends.jev import (
    MAX_CHOICE_OPTIONS,
    NO_ROUTE,
    NOT_STATED,
    ROUTE_QID,
    ArgSlot,
    build_questions,
    build_slots,
    candidates,
    classify,
    coerce,
)

CANCEL_TOOL = {
    "type": "function",
    "function": {
        "name": "cancel_order",
        "description": "Cancel a customer's order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["order_id"],
        },
    },
}
PRODUCTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_products",
        "description": "Search products by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
    },
}
TAGS_TOOL = {
    "type": "function",
    "function": {
        "name": "tag_item",
        "description": "Tag an item.",
        "parameters": {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    },
}
TOOLS = [CANCEL_TOOL, PRODUCTS_TOOL]


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "integer"}, ("integer", ())),
        ({"type": "number"}, ("number", ())),
        ({"type": "boolean"}, ("boolean", ())),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, ("string", ())),
        ({"enum": ["s", "m"], "type": "string"}, ("enum", ("s", "m"))),
        ({"type": "array", "items": {"type": "string"}}, (None, ())),
        ({"type": "object"}, (None, ())),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, (None, ())),
    ],
)
def test_classify_shapes(schema, expected):
    assert classify(schema, {}) == expected


def test_classify_resolves_local_and_root_defs():
    color = {"enum": ["red", "blue"], "type": "string"}
    local = {
        "$defs": {"Color": color},
        "anyOf": [{"$ref": "#/$defs/Color"}, {"type": "null"}],
    }
    assert classify(local, {}) == ("enum", ("red", "blue"))
    assert classify({"$ref": "#/$defs/Color"}, {"Color": color}) == ("enum", ("red", "blue"))


def test_classify_dangling_ref_is_unsupported():
    assert classify({"$ref": "#/$defs/Missing"}, {}) == (None, ())


def test_candidates_integer_strips_punctuation():
    assert candidates("cancel my order #4521, now", "integer", (), 6) == ("4521",)


def test_candidates_number_includes_decimals():
    assert candidates("under 19.99 or 20", "number", (), 6) == ("19.99", "20")


def test_candidates_string_ngrams_are_bounded_and_deduplicated():
    assert candidates("red red shoe", "string", (), 2) == ("red", "red red", "red shoe", "shoe")


def test_candidates_enum_and_boolean():
    assert candidates("anything", "enum", ("s", 2), 6) == ("s", "2")
    assert candidates("anything", "boolean", (), 6) == ("true", "false")


def test_candidates_cap_leaves_room_for_not_stated():
    query = " ".join(f"w{i}" for i in range(100))
    assert len(candidates(query, "string", (), 6)) == MAX_CHOICE_OPTIONS - 1


def test_candidates_drop_literal_not_stated():
    assert NOT_STATED not in candidates("reason not_stated", "string", (), 1)


def test_build_slots_covers_every_route_param():
    slots = build_slots("cancel order 123", TOOLS, 6)
    assert [(s.qid, s.route, s.param, s.kind, s.required) for s in slots] == [
        ("arg0", "cancel_order", "order_id", "integer", True),
        ("arg1", "cancel_order", "reason", "string", False),
        ("arg2", "list_products", "category", "string", False),
        ("arg3", "list_products", "limit", "integer", False),
    ]
    assert slots[0].options == ("123",)


def test_build_slots_marks_unsupported_types():
    slots = build_slots("tag it red", [TAGS_TOOL], 6)
    assert (slots[0].kind, slots[0].options, slots[0].required) == (None, (), True)


def test_build_questions_asks_route_and_slots_with_candidates():
    slots = build_slots("show products", [PRODUCTS_TOOL], 6)
    questions = build_questions([PRODUCTS_TOOL], slots)
    # `limit` has no digits to choose from, so it gets no question.
    assert set(questions) == {ROUTE_QID, "arg0"}
    assert set(questions[ROUTE_QID].criteria) == {"list_products", NO_ROUTE}
    assert NOT_STATED in questions["arg0"].criteria


def test_build_questions_skips_unsupported_types():
    slots = build_slots("tag it red", [TAGS_TOOL], 6)
    assert set(build_questions([TAGS_TOOL], slots)) == {ROUTE_QID}


def _slot(kind, enum_values=()):
    return ArgSlot(
        qid="arg0",
        route="r",
        param="p",
        kind=kind,
        required=False,
        options=(),
        enum_values=enum_values,
    )


@pytest.mark.parametrize(
    ("kind", "raw", "expected"),
    [
        ("integer", "4521", 4521),
        ("integer", "4.5", None),
        ("number", "19.99", 19.99),
        ("boolean", "true", True),
        ("boolean", "false", False),
        ("string", "wrong size", "wrong size"),
    ],
)
def test_coerce(kind, raw, expected):
    assert coerce(_slot(kind), raw) == expected


def test_coerce_enum_maps_back_to_original_value():
    assert coerce(_slot("enum", (1, 2)), "2") == 2
    assert coerce(_slot("enum", ("s",)), "xl") is None
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_backends_jev.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'fastapi_ai_router.backends.jev'`.

- [ ] **Step 4: Implement the helpers**

Create `src/fastapi_ai_router/backends/jev.py`:

```python
"""JevBackend: route requests with TypeSafe's Jev classifier instead of a generative LLM.

Jev picks from options; it never writes text. The route is a Choice over tool
names, and each argument is a Choice over candidate values cut from the query
(numbers, word spans, enum values). Every question goes out in one request, and
code reads the answers for the route Jev picked. Install with
`pip install fastapi-ai-router[jev]`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

try:
    from typesafe_sdk import Choice
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "JevBackend requires the 'jev' extra. "
        "Install with: pip install fastapi-ai-router[jev]"
    ) from exc

from fastapi_ai_router.backends import ToolDef

NOT_STATED = "not_stated"
NO_ROUTE = "none_of_the_above"
ROUTE_QID = "route"
MAX_CHOICE_OPTIONS = 255  # the API rejects more with 400 "Too many choices"
ROUTE_INSTRUCTIONS = "Which API route should handle this user request?"

ParamKind = Literal["integer", "number", "boolean", "string", "enum"]
_SCALAR_KINDS = frozenset({"integer", "number", "boolean", "string"})

_WORD = re.compile(r"[\w'-]+")
_INT = re.compile(r"\d+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ArgSlot:
    """One (route, param) pair and the candidate values offered to Jev."""

    qid: str
    route: str
    param: str
    kind: ParamKind | None  # None: a type Jev can't extract (object, array, ...)
    required: bool
    options: tuple[str, ...]
    enum_values: tuple[Any, ...] = ()


def _resolve(
    schema: Mapping[str, Any], root_defs: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Unwrap Optional (anyOf with null) and a local $ref. None if unresolvable."""
    defs = {**root_defs, **(schema.get("$defs") or {})}
    variants = schema.get("anyOf")
    if isinstance(variants, list):
        non_null = [v for v in variants if v.get("type") != "null"]
        if len(non_null) != 1:
            return None
        schema = non_null[0]
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = defs.get(ref.removeprefix("#/$defs/"))
        return target if isinstance(target, Mapping) else None
    return schema


def classify(
    schema: Mapping[str, Any], root_defs: Mapping[str, Any]
) -> tuple[ParamKind | None, tuple[Any, ...]]:
    """What Jev can extract for a param's JSON schema, plus its enum values."""
    resolved = _resolve(schema, root_defs)
    if resolved is None:
        return None, ()
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        return "enum", tuple(enum)
    kind = resolved.get("type")
    if kind in _SCALAR_KINDS:
        return kind, ()
    return None, ()


def candidates(
    query: str, kind: ParamKind, enum_values: tuple[Any, ...], max_span_words: int
) -> tuple[str, ...]:
    """Candidate values for one param, capped to fit a single Choice."""
    if kind == "enum":
        spans = [str(v) for v in enum_values]
    elif kind == "boolean":
        spans = ["true", "false"]
    elif kind == "integer":
        spans = _INT.findall(query)
    elif kind == "number":
        spans = _NUMBER.findall(query)
    else:
        words = _WORD.findall(query)
        spans = [
            " ".join(words[i:j])
            for i in range(len(words))
            for j in range(i + 1, min(len(words), i + max_span_words) + 1)
        ]
    unique = [s for s in dict.fromkeys(spans) if s != NOT_STATED]
    # ponytail: long queries lose their late spans at the cap; extract per route
    # in a second call if that bites.
    return tuple(unique[: MAX_CHOICE_OPTIONS - 1])


def build_slots(query: str, tools: list[ToolDef], max_span_words: int) -> list[ArgSlot]:
    """One slot per (route, param) across every tool."""
    slots: list[ArgSlot] = []
    for tool in tools:
        params = tool["function"]["parameters"]
        root_defs = params.get("$defs") or {}
        required = set(params.get("required") or [])
        for name, schema in (params.get("properties") or {}).items():
            kind, enum_values = classify(schema, root_defs)
            options = candidates(query, kind, enum_values, max_span_words) if kind else ()
            slots.append(
                ArgSlot(
                    qid=f"arg{len(slots)}",
                    route=tool["function"]["name"],
                    param=name,
                    kind=kind,
                    required=name in required,
                    options=options,
                    enum_values=enum_values,
                )
            )
    return slots


def _arg_instructions(slot: ArgSlot) -> str:
    prefix = f"If this request is handled by route `{slot.route}`,"
    if slot.kind in ("enum", "boolean"):
        return f"{prefix} which option matches the value it gives for parameter `{slot.param}`?"
    return (
        f"{prefix} which exact words of the request are the value of "
        f"parameter `{slot.param}` ({slot.kind})?"
    )


def build_questions(tools: list[ToolDef], slots: list[ArgSlot]) -> dict[str, Choice]:
    """The route question plus one question per slot that has candidates."""
    # ponytail: asks every route's params up front (one round trip); split into
    # route-then-args calls if large apps hit Jev's 64k-token context limit.
    routes: dict[str, str | None] = {
        t["function"]["name"]: t["function"]["description"] or None for t in tools
    }
    routes[NO_ROUTE] = "The request fits none of the other routes."
    questions = {ROUTE_QID: Choice(instructions=ROUTE_INSTRUCTIONS, criteria=routes)}
    for slot in slots:
        if slot.options:
            criteria: dict[str, str | None] = dict.fromkeys(slot.options)
            criteria[NOT_STATED] = "The request does not state this value."
            questions[slot.qid] = Choice(
                instructions=_arg_instructions(slot), criteria=criteria
            )
    return questions


def coerce(slot: ArgSlot, raw: str) -> Any:
    """Turn Jev's chosen option into a typed value; None if it doesn't convert."""
    try:
        if slot.kind == "integer":
            return int(raw)
        if slot.kind == "number":
            return float(raw)
    except ValueError:
        return None
    if slot.kind == "boolean":
        return raw == "true"
    if slot.kind == "enum":
        return {str(v): v for v in slot.enum_values}.get(raw)
    return raw
```

- [ ] **Step 5: Run the tests, lint, and types**

Run: `uv run pytest tests/unit/test_backends_jev.py -v && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: 27 new tests PASS; ruff and mypy clean; full suite `106 passed, 2 skipped`. If `ruff format --check` flags the new files, run `uv run ruff format src/fastapi_ai_router/backends/jev.py tests/unit/test_backends_jev.py` and re-run.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/fastapi_ai_router/backends/jev.py tests/unit/test_backends_jev.py
git commit -m "feat(jev): add span-selection helpers and the jev extra"
```

---

### Task 6: `JevBackend.call`: decision logic, fallback, errors

**Files:**
- Modify: `src/fastapi_ai_router/backends/jev.py`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_backends_jev.py`

**Interfaces:**
- Consumes: everything Task 5 produces; `ToolCall.confidence` (Task 4); `LLMBackend`, `Message`, `ToolCall` from `fastapi_ai_router.backends`; `LLMBackendError`; `FakeLLMBackend` (`returns=`, `.calls[i].tools`).
- Produces:
  - `@dataclass class JevBackend(LLMBackend)` with fields `model: str = "jev-latest"`, `api_key: str | None = None`, `min_confidence: float = 0.5`, `fallback: LLMBackend | None = None`, `max_span_words: int = 6`, `client: AsyncTypeSafeClient | None = None`, and `async def call(self, messages: list[Message], tools: list[ToolDef]) -> ToolCall | None`
  - `collect_args(response: SystemOneResponse, slots: list[ArgSlot]) -> tuple[dict[str, Any], bool]`
  - `tests.conftest.JevStubClient(answers: dict[str, tuple[str, float]] | BaseException)`, with `.calls: list[tuple[state, questions]]`, returning model `"jev-1.13.0"`, `input_tokens=100`, `output_tokens=10`. Task 7 uses it.

- [ ] **Step 1: Add the stub client to conftest**

In `tests/conftest.py`, add `from typesafe_sdk import SystemOneResponse` to the third-party imports, then append:

```python
class JevStubClient:
    """Stands in for typesafe_sdk.AsyncTypeSafeClient: canned Choice answers, records requests."""

    def __init__(self, answers: dict[str, tuple[str, float]] | BaseException) -> None:
        self.answers = answers
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        self.calls.append((state, questions))
        if isinstance(self.answers, BaseException):
            raise self.answers
        return SystemOneResponse.model_validate(
            {
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    qid: {
                        "type": "choice",
                        "choice": choice,
                        "confidence": conf,
                        "probabilities": {choice: conf},
                    }
                    for qid, (choice, conf) in self.answers.items()
                },
            }
        )
```

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_backends_jev.py`, replace the import block at the top with:

```python
import pytest
from typesafe_sdk import TypeSafeError

from fastapi_ai_router.backends import ToolCall, jev
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.backends.jev import (
    MAX_CHOICE_OPTIONS,
    NO_ROUTE,
    NOT_STATED,
    ROUTE_QID,
    ArgSlot,
    JevBackend,
    build_questions,
    build_slots,
    candidates,
    classify,
    coerce,
)
from fastapi_ai_router.errors import LLMBackendError
from tests.conftest import JevStubClient
```

Then append:

```python
def _messages(query):
    return [
        {"role": "system", "content": "You are a router."},
        {"role": "user", "content": query},
    ]


async def test_call_routes_and_extracts_args_in_one_request():
    stub = JevStubClient(
        {
            ROUTE_QID: ("cancel_order", 0.98),
            "arg0": ("123", 1.0),
            "arg1": ("because it was a duplicate", 0.6),
        }
    )
    query = "cancel order 123 because it was a duplicate"
    result = await JevBackend(client=stub).call(_messages(query), TOOLS)
    assert result == ToolCall(
        name="cancel_order",
        args={"order_id": 123, "reason": "because it was a duplicate"},
        reasoning=None,
        prompt_tokens=100,
        completion_tokens=10,
        model="jev-1.13.0",
        confidence=0.98,
    )
    assert len(stub.calls) == 1
    state, questions = stub.calls[0]
    assert state == query
    assert set(questions) == {ROUTE_QID, "arg0", "arg1", "arg2", "arg3"}


async def test_call_omits_optional_not_stated_args():
    stub = JevStubClient(
        {ROUTE_QID: ("cancel_order", 1.0), "arg0": ("77", 1.0), "arg1": (NOT_STATED, 1.0)}
    )
    result = await JevBackend(client=stub).call(_messages("cancel order 77"), TOOLS)
    assert result is not None
    assert result.args == {"order_id": 77}


async def test_confident_no_match_returns_none_without_calling_fallback():
    fallback = FakeLLMBackend(returns=None)
    stub = JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})
    backend = JevBackend(client=stub, fallback=fallback)
    assert await backend.call(_messages("weather in Paris"), TOOLS) is None
    assert fallback.calls == []


async def test_low_confidence_without_fallback_returns_none():
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.3)})
    assert await JevBackend(client=stub).call(_messages("hmm"), TOOLS) is None


async def test_low_confidence_hands_all_tools_to_fallback():
    llm_call = ToolCall(
        name="list_products",
        args={},
        reasoning="r",
        prompt_tokens=5,
        completion_tokens=1,
        model="llm",
    )
    fallback = FakeLLMBackend(returns=llm_call)
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.3)})
    result = await JevBackend(client=stub, fallback=fallback).call(_messages("hmm"), TOOLS)
    assert result == llm_call
    assert fallback.calls[0].tools == TOOLS


async def test_missing_required_arg_narrows_fallback_to_chosen_route():
    llm_call = ToolCall(
        name="cancel_order",
        args={"order_id": 5},
        reasoning=None,
        prompt_tokens=5,
        completion_tokens=1,
        model="llm",
    )
    fallback = FakeLLMBackend(returns=llm_call)
    # "cancel my order" has no digits, so order_id gets no question at all.
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9), "arg1": (NOT_STATED, 1.0)})
    backend = JevBackend(client=stub, fallback=fallback)
    result = await backend.call(_messages("cancel my order"), TOOLS)
    assert fallback.calls[0].tools == [CANCEL_TOOL]
    assert result is not None
    assert (result.model, result.confidence) == ("llm", 0.9)


async def test_missing_required_arg_without_fallback_returns_partial_call():
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9), "arg1": (NOT_STATED, 1.0)})
    result = await JevBackend(client=stub).call(_messages("cancel my order"), TOOLS)
    assert result is not None
    assert (result.name, result.args) == ("cancel_order", {})


async def test_unsupported_required_param_goes_to_fallback():
    fallback = FakeLLMBackend(returns=None)
    stub = JevStubClient({ROUTE_QID: ("tag_item", 0.95)})
    await JevBackend(client=stub, fallback=fallback).call(_messages("tag it red"), [TAGS_TOOL])
    assert fallback.calls[0].tools == [TAGS_TOOL]


async def test_sdk_errors_become_llm_backend_errors():
    stub = JevStubClient(TypeSafeError("rate limited"))
    with pytest.raises(LLMBackendError) as ei:
        await JevBackend(client=stub).call(_messages("cancel order 1"), TOOLS)
    assert isinstance(ei.value.upstream, TypeSafeError)


async def test_missing_expected_answer_raises():
    # order_id had a candidate ("1"), so an answer for arg0 was expected.
    stub = JevStubClient({ROUTE_QID: ("cancel_order", 0.9)})
    with pytest.raises(LLMBackendError, match="arg0"):
        await JevBackend(client=stub).call(_messages("cancel order 1"), TOOLS)


async def test_client_is_created_once_from_settings(monkeypatch):
    made = []

    def factory(**kwargs):
        made.append(kwargs)
        return JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})

    monkeypatch.setattr(jev, "AsyncTypeSafeClient", factory)
    backend = JevBackend(api_key="k", model="jev-1.13.0")
    await backend.call(_messages("hi"), TOOLS)
    await backend.call(_messages("hi"), TOOLS)
    assert made == [{"api_key": "k", "model": "jev-1.13.0"}]
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_backends_jev.py -v`
Expected: collection error, `ImportError: cannot import name 'JevBackend'`.

- [ ] **Step 4: Implement**

In `src/fastapi_ai_router/backends/jev.py`, replace the import section (from `from __future__ import annotations` through `from fastapi_ai_router.backends import ToolDef`) with:

```python
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        ChoiceAnswer,
        SystemOneResponse,
        TypeSafeError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "JevBackend requires the 'jev' extra. "
        "Install with: pip install fastapi-ai-router[jev]"
    ) from exc

from fastapi_ai_router.backends import LLMBackend, Message, ToolCall, ToolDef
from fastapi_ai_router.errors import LLMBackendError
```

Append to the end of the file:

```python
def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def _answer(response: SystemOneResponse, qid: str) -> ChoiceAnswer:
    answer = response.choices.get(qid)
    if answer is None:
        raise LLMBackendError(f"Jev response is missing answer {qid!r}")
    return answer


def collect_args(
    response: SystemOneResponse, slots: list[ArgSlot]
) -> tuple[dict[str, Any], bool]:
    """Typed args for one route, and whether a required param came back empty."""
    args: dict[str, Any] = {}
    missing_required = False
    for slot in slots:
        value = None
        if slot.options:
            choice = _answer(response, slot.qid).choice
            value = None if choice == NOT_STATED else coerce(slot, choice)
        if value is None:
            missing_required = missing_required or slot.required
        else:
            args[slot.param] = value
    return args, missing_required


@dataclass
class JevBackend(LLMBackend):
    """Routes with Jev in one request; optionally hands the hard cases to an LLM.

    `fallback` runs only when Jev is unsure of the route (it gets every tool) or
    the chosen route needs a value Jev couldn't supply (it gets that route's tool
    only). With no fallback, the router never calls an LLM.
    """

    model: str = "jev-latest"
    api_key: str | None = None  # None: the SDK reads TYPESAFE_API_KEY
    min_confidence: float = 0.5
    fallback: LLMBackend | None = None
    max_span_words: int = 6
    client: AsyncTypeSafeClient | None = None

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None:
        query = _last_user_content(messages)
        slots = build_slots(query, tools, self.max_span_words)
        response = await self._ask(query, build_questions(tools, slots))
        route = _answer(response, ROUTE_QID)
        if route.confidence < self.min_confidence:
            return await self._fall_back(messages, tools, confidence=None)
        if route.choice == NO_ROUTE:
            return None
        route_slots = [s for s in slots if s.route == route.choice]
        args, missing_required = collect_args(response, route_slots)
        if missing_required and self.fallback is not None:
            narrowed = [t for t in tools if t["function"]["name"] == route.choice]
            return await self._fall_back(messages, narrowed, confidence=route.confidence)
        return ToolCall(
            name=route.choice,
            args=args,
            reasoning=None,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model=response.model,
            confidence=route.confidence,
        )

    async def _ask(self, query: str, questions: dict[str, Choice]) -> SystemOneResponse:
        try:
            if self.client is None:
                # ponytail: one shared client, never closed; AIRouter has no shutdown hook.
                self.client = AsyncTypeSafeClient(api_key=self.api_key, model=self.model)
            return await self.client.system_one(query, questions)
        except TypeSafeError as exc:
            raise LLMBackendError(f"Jev call failed: {exc}", upstream=exc) from exc

    async def _fall_back(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        *,
        confidence: float | None,
    ) -> ToolCall | None:
        if self.fallback is None:
            return None
        result = await self.fallback.call(messages, tools)
        if result is None or confidence is None:
            return result
        # Jev still chose the route; only the argument filling was delegated.
        return dataclasses.replace(result, confidence=confidence)


__all__ = ["JevBackend"]
```

- [ ] **Step 5: Run tests, lint, types**

Run: `uv run ruff check --fix src tests && uv run ruff format src tests && uv run mypy && uv run pytest -q`
Expected: ruff clean; mypy `Success`; `117 passed, 2 skipped`.

Pre-checked: `ChoiceAnswer.choice` is `str` and `ChoiceAnswer.confidence` is `float`, so `ToolCall(name=route.choice, ...)` type-checks. The SDK ships `py.typed`, and mypy 2.3.1 `--strict` passes on this call pattern.

- [ ] **Step 6: Commit**

```bash
git add src/fastapi_ai_router/backends/jev.py tests/conftest.py tests/unit/test_backends_jev.py
git commit -m "feat(jev): add JevBackend with confidence gate and optional LLM fallback"
```

---

### Task 7: Integration and gated end-to-end tests

**Files:**
- Create: `tests/integration/test_jev_backend.py`
- Create: `tests/e2e/test_with_real_jev.py`

**Interfaces:**
- Consumes: `JevBackend`, `ROUTE_QID`, `NOT_STATED` (Tasks 5 and 6); `JevStubClient` (Task 6); `sample_app` fixture (existing: `cancel` route with path `order_id` + body `reason` + auth header `Bearer good`; `list_products` with `category`, `limit`); `missing_path_param` response (Task 3); `Decision.confidence` (Task 4).
- Produces: tests only.

- [ ] **Step 1: Write the integration tests**

Create `tests/integration/test_jev_backend.py`:

```python
from fastapi.testclient import TestClient

from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.fake import FakeLLMBackend
from fastapi_ai_router.backends.jev import NO_ROUTE, NOT_STATED, ROUTE_QID, JevBackend
from fastapi_ai_router.core import AIRouter
from fastapi_ai_router.observability import Decision
from tests.conftest import JevStubClient

AUTH = {"Authorization": "Bearer good"}


def test_jev_backend_routes_and_dispatches(sample_app):
    decisions: list[Decision] = []

    async def hook(d: Decision) -> None:
        decisions.append(d)

    # sample_app slots: cancel -> arg0 order_id, arg1 reason; list_products -> arg2, arg3
    stub = JevStubClient(
        {ROUTE_QID: ("cancel", 0.97), "arg0": ("7", 1.0), "arg1": ("duplicate", 0.8)}
    )
    AIRouter(sample_app, llm=JevBackend(client=stub), on_decision=hook)
    resp = TestClient(sample_app).post(
        "/ai", json={"query": "cancel order 7 as a duplicate"}, headers=AUTH
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["args"] == {"order_id": 7, "reason": "duplicate"}
    assert body["result"]["status"] == "cancelled"
    assert decisions[0].confidence == 0.97


def test_jev_backend_missing_order_id_returns_422(sample_app):
    stub = JevStubClient({ROUTE_QID: ("cancel", 0.9), "arg1": (NOT_STATED, 1.0)})
    AIRouter(sample_app, llm=JevBackend(client=stub))
    resp = TestClient(sample_app).post(
        "/ai", json={"query": "cancel my order"}, headers=AUTH
    )
    assert resp.status_code == 422
    assert resp.json()["missing"] == ["order_id"]


def test_jev_backend_no_match_returns_422(sample_app):
    stub = JevStubClient({ROUTE_QID: (NO_ROUTE, 1.0)})
    AIRouter(sample_app, llm=JevBackend(client=stub))
    resp = TestClient(sample_app).post("/ai", json={"query": "what's the weather"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"


def test_jev_backend_low_confidence_uses_fallback(sample_app):
    fallback = FakeLLMBackend(
        returns=ToolCall(
            name="list_products",
            args={"category": "books"},
            reasoning="llm picked",
            prompt_tokens=5,
            completion_tokens=1,
            model="llm",
        )
    )
    stub = JevStubClient({ROUTE_QID: ("cancel", 0.2)})
    AIRouter(sample_app, llm=JevBackend(client=stub, fallback=fallback))
    resp = TestClient(sample_app).post("/ai", json={"query": "books please"})
    assert resp.status_code == 200
    assert resp.json()["args"] == {"category": "books"}
```

- [ ] **Step 2: Run the integration tests**

Run: `uv run pytest tests/integration/test_jev_backend.py -v`
Expected: 4 PASS. If they fail, report the failure output verbatim. Tasks 3 through 6 already provide all the behavior, so a failure here means a bug in one of them. Fix it at its source.

- [ ] **Step 3: Write the gated end-to-end tests**

Create `tests/e2e/test_with_real_jev.py`:

```python
"""Real Jev smoke tests. Gated behind RUN_LLM_TESTS=1 and TYPESAFE_API_KEY.

Run: RUN_LLM_TESTS=1 uv run --env-file .env pytest tests/e2e/test_with_real_jev.py -v
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.jev import JevBackend

pytestmark = pytest.mark.e2e

CANCEL = "POST /orders/{order_id}/cancel"
PRODUCTS = "GET /products"


@pytest.fixture(autouse=True)
def gate() -> None:
    if os.environ.get("RUN_LLM_TESTS") != "1":
        pytest.skip("RUN_LLM_TESTS=1 not set")
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY not set")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    @ai_route(description="Cancel a customer's order and record a reason.")
    def cancel_order(order_id: int, reason: str | None = None) -> dict:
        return {"status": "cancelled", "order_id": order_id, "reason": reason}

    @app.get("/products")
    @ai_route(description="Search products by category with an optional limit.")
    def list_products(category: str | None = None, limit: int = 20) -> dict:
        return {"items": [], "category": category, "limit": limit}

    AIRouter(app, llm=JevBackend())
    return TestClient(app)


@pytest.mark.parametrize(
    ("query", "endpoint", "expected_args"),
    [
        ("cancel order 123 because it was a duplicate", CANCEL, {"order_id": 123}),
        ("cancel my order #4521, I ordered the wrong size", CANCEL, {"order_id": 4521}),
        ("cancel order 77", CANCEL, {"order_id": 77}),
        ("show me 5 laptops", PRODUCTS, {"category": "laptops", "limit": 5}),
        ("list 10 items in the kitchen category", PRODUCTS, {"category": "kitchen", "limit": 10}),
    ],
)
def test_jev_routes_and_extracts(client, query, endpoint, expected_args):
    resp = client.post("/ai", json={"query": query})
    assert resp.status_code == 200
    body = resp.json()
    assert body["endpoint"] == endpoint
    assert {k: body["args"].get(k) for k in expected_args} == expected_args


def test_jev_extracts_free_text_reason(client):
    body = client.post(
        "/ai", json={"query": "cancel order 123 because it was a duplicate"}
    ).json()
    assert "duplicate" in body["args"]["reason"]


def test_jev_leaves_unstated_optionals_out(client):
    body = client.post("/ai", json={"query": "show me products"}).json()
    assert body["endpoint"] == PRODUCTS
    assert body["args"] == {}


def test_jev_no_route_for_off_topic_query(client):
    resp = client.post("/ai", json={"query": "what's the weather in Paris"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_route_matched"
```

- [ ] **Step 4: Verify the gate skips by default, then run for real**

Run: `uv run pytest tests/e2e/test_with_real_jev.py -q`
Expected: `8 skipped`.

Run: `RUN_LLM_TESTS=1 uv run --env-file .env pytest tests/e2e/test_with_real_jev.py -v`
Expected: `8 passed`. These are real Jev calls costing under $0.001 in total. A failure is a real finding: paste the full output into your report. Do not loosen the assertions to make them pass.

- [ ] **Step 5: Full suite, lint, types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: all clean; `121 passed, 10 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_jev_backend.py tests/e2e/test_with_real_jev.py
git commit -m "test(jev): add stubbed integration and gated real-Jev tests"
```

---

### Task 8: Docs: README, CHANGELOG, example

**Files:**
- Modify: `README.md` (Install, new Jev section, Error semantics table, test counts, Testing block)
- Modify: `CHANGELOG.md`
- Create: `examples/05_jev.py`
- Modify: `examples/README.md`

**Interfaces:**
- Consumes: the finished feature.
- Produces: documentation only.

- [ ] **Step 1: README Install block**

In the `## Install` section, replace the first code block:

```bash
pip install fastapi-ai-router[litellm]
```

with:

```bash
pip install fastapi-ai-router[litellm]   # any LLM, via LiteLLM
pip install fastapi-ai-router[jev]       # TypeSafe's Jev classifier (see below)
```

- [ ] **Step 2: README Jev section**

Insert this section after the `## Bring your own LLM` section's closing `---` and before `## Observability`:

````markdown
## Jev backend: routing without an LLM

[Jev](https://docs.typesafe.ai) is TypeSafe's classifier. It doesn't generate text; it picks among options you give it and reports a calibrated confidence. `JevBackend` routes with that. The route is a choice over your tools, and each argument is a choice over values cut from the query (numbers, word spans, enum values). One request covers the route and every argument, typically in 120–350 ms and for a few thousandths of a cent.

```python
from fastapi_ai_router.backends.jev import JevBackend

AIRouter(app, llm=JevBackend())  # reads TYPESAFE_API_KEY
```

Jev handles integers, numbers, booleans, enums, and strings that appear word for word in the query. It can't build lists or nested objects, and it doesn't read number words ("five"). For those cases, and whenever it isn't sure which route fits, you can hand the request to an LLM:

```python
from fastapi_ai_router.backends.litellm import LiteLLMBackend

AIRouter(app, llm=JevBackend(fallback=LiteLLMBackend(model="gpt-4o-mini")))
```

| Jev's answer | No fallback | With `fallback=` |
|---|---|---|
| Confident that no route fits | 422 `no_route_matched` | same; the LLM is not called |
| Route confidence below `min_confidence` (default 0.5) | 422 `no_route_matched` | the LLM sees every tool |
| Route chosen, a required argument missing | 422 naming the missing field | the LLM sees only that route's tool |

Your `on_decision` hook receives Jev's route confidence as `Decision.confidence`.

---
````

- [ ] **Step 3: README error table and test counts**

In the `## Error semantics` table, add this row after the `UnknownTool` row:

```markdown
| `MissingPathParams` (backend left out a path argument) | 422 | `{"error":"missing_path_param", "missing":[…], "endpoint":"…"}` |
```

Replace `The whole test suite uses `FakeLLMBackend` — **74 tests pass deterministically without a single API key.**` with:

```markdown
The whole test suite uses `FakeLLMBackend` and a stubbed Jev client, so **it passes deterministically without a single API key.**
```

In the `## Testing` code block, replace `uv run pytest                                   # 74 tests, deterministic, no API keys` with `uv run pytest                                   # deterministic, no API keys`, and add this line at the end of the block:

```bash
RUN_LLM_TESTS=1 uv run --env-file .env pytest tests/e2e/test_with_real_jev.py   # real Jev calls
```

- [ ] **Step 4: CHANGELOG**

In `CHANGELOG.md`, insert above `## [0.1.0] — 2026-04-25`:

```markdown
## [Unreleased]

### Added
- `JevBackend` (`fastapi-ai-router[jev]`): routes with TypeSafe's Jev classifier in a single request, with no LLM call. An optional `fallback=` backend handles low route confidence and arguments Jev can't extract.
- `confidence` field on `ToolCall` and `Decision`, defaulting to `None`.
- `MissingPathParams` error.

### Fixed
- A backend that leaves out a path argument now gets `422 missing_path_param` instead of `500 dispatch_error`.
- Enum fields on flattened body models keep their `$defs`, so their `$ref`s resolve.

### Changed
- Dependencies refreshed. Dev-tool floors raised to ruff 0.16, mypy 2.3, pytest 9.1.

```

- [ ] **Step 5: Example**

Create `examples/05_jev.py`:

```python
"""The basic example, routed by TypeSafe's Jev instead of an LLM.

Run:
    uv run --env-file .env uvicorn examples.05_jev:app --reload

Then:
    curl -X POST localhost:8000/ai -H "content-type: application/json" \
         -d '{"query":"cancel order 123 because it was a duplicate"}'
"""

from fastapi import FastAPI

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends.jev import JevBackend

app = FastAPI()


@app.post("/orders/{order_id}/cancel")
@ai_route(description="Cancel a customer's order and record a reason.")
def cancel_order(order_id: int, reason: str | None = None) -> dict:
    return {"status": "cancelled", "order_id": order_id, "reason": reason}


@app.get("/products")
@ai_route(description="Search products by category with an optional limit.")
def list_products(category: str | None = None, limit: int = 20) -> dict:
    return {"items": [], "category": category, "limit": limit}


# Reads TYPESAFE_API_KEY. Pass fallback=LiteLLMBackend(...) to cover what Jev can't extract.
AIRouter(app, llm=JevBackend())
```

Add this row to the table in `examples/README.md`:

```markdown
| `05_jev.py` | Routing with TypeSafe's Jev classifier instead of an LLM (needs `TYPESAFE_API_KEY`). |
```

- [ ] **Step 6: Verify the example against the real API**

Run:

```bash
uv run --env-file .env python -c "import importlib; from fastapi.testclient import TestClient; app = importlib.import_module('examples.05_jev').app; print(TestClient(app).post('/ai', json={'query': 'show me 5 laptops'}).json())"
```

Expected: a dict with `'endpoint': 'GET /products'` and `'args': {'category': 'laptops', 'limit': 5}`.

- [ ] **Step 7: Prose check and format**

Run: `uv run --no-project python ~/.agents/skills/humanizer-stack/scripts/copy_scan.py README.md && uv run --no-project python ~/.agents/skills/humanizer-stack/scripts/copy_scan.py CHANGELOG.md && uv run ruff format --check .`
Expected: `clean` for both files; ruff reports everything formatted. If ruff wants to reformat the new README code blocks, run `uv run ruff format README.md` and re-check.

- [ ] **Step 8: Commit**

```bash
git add README.md CHANGELOG.md examples/05_jev.py examples/README.md
git commit -m "docs: document JevBackend, missing_path_param, and the 05_jev example"
```

---

### Task 9: Final verification and review

**Files:** none unless review findings require fixes.

- [ ] **Step 1: All gates plus coverage**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov=fastapi_ai_router --cov-report=term-missing -q`
Expected: everything clean; coverage at or above 80% (the `fail_under` gate); `jev.py` coverage at 90% or higher, with only the `ImportError` guard uncovered.

- [ ] **Step 2: Real Jev pass, once more on the final tree**

Run: `RUN_LLM_TESTS=1 uv run --env-file .env pytest tests/e2e/test_with_real_jev.py -q`
Expected: `8 passed`.

- [ ] **Step 3: Secrets check**

Run: `KEY=$(sed -n 's/^TYPESAFE_API_KEY=//p' .env); git log -p main..HEAD | grep -cF "$KEY"; git check-ignore .env`
Expected: `0` (the key appears in no commit), then `.env`. This prints only a count, never the key.

- [ ] **Step 4: Code review**

Run `/code-review` on `main..HEAD` and dispatch the `code-reviewer` subagent on the same range. Fix every CRITICAL and HIGH finding, each in its own `fix:` commit with a test, and re-run Step 1 after the fixes.

- [ ] **Step 5: Report**

Summarize for the user: commits on `feat/jev-backend`, the test count, coverage, the e2e result, the review findings and how each was handled. Do not merge, push, or tag; that waits for the user.
