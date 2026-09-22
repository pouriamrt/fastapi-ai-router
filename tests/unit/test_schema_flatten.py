"""Tests for OpenAPI → RouteSpec projection."""

from enum import StrEnum

from fastapi import Body, FastAPI
from pydantic import BaseModel

from fastapi_ai_router.schema import route_to_spec


def _route_named(app: FastAPI, name: str):
    for r in app.routes:
        if getattr(r, "name", None) == name:
            return r
    raise AssertionError(f"route {name} not found")


def test_path_only_route():
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel", name="cancel_order")
    def cancel_order(order_id: int) -> dict:
        return {"ok": True}

    spec = route_to_spec(_route_named(app, "cancel_order"))
    assert spec is not None

    assert spec.name == "cancel_order"
    assert spec.method == "POST"
    assert spec.path_template == "/orders/{order_id}/cancel"
    assert spec.param_locations == {"order_id": "path"}
    props = spec.parameters_schema["properties"]
    assert props["order_id"]["type"] == "integer"
    assert "order_id" in spec.parameters_schema["required"]


def test_query_only_route():
    app = FastAPI()

    @app.get("/products", name="list_products")
    def list_products(category: str | None = None, limit: int = 20) -> dict:
        return {"items": []}

    spec = route_to_spec(_route_named(app, "list_products"))
    assert spec is not None

    assert spec.method == "GET"
    assert spec.param_locations == {"category": "query", "limit": "query"}
    assert spec.parameters_schema.get("required", []) == []


def test_body_pydantic_model_route():
    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float

    @app.post("/items", name="create_item")
    def create_item(item: Item) -> dict:
        return {"ok": True}

    spec = route_to_spec(_route_named(app, "create_item"))
    assert spec is not None

    assert spec.param_locations == {"name": "body", "price": "body"}
    props = spec.parameters_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["price"]["type"] == "number"


def test_combined_path_query_body_route():
    app = FastAPI()

    class Order(BaseModel):
        reason: str

    @app.post("/orders/{order_id}/cancel", name="cancel_order_full")
    def cancel_order_full(order_id: int, dry_run: bool = False, order: Order | None = None):
        return {}

    spec = route_to_spec(_route_named(app, "cancel_order_full"))
    assert spec is not None

    assert spec.param_locations["order_id"] == "path"
    assert spec.param_locations["dry_run"] == "query"
    assert "reason" in spec.param_locations or "order" in spec.param_locations


def test_lone_list_body_is_sent_whole():
    app = FastAPI()

    @app.post("/bulk", name="bulk_create")
    def bulk_create(items: list[str] = Body(...)) -> dict:  # noqa: B008
        return {"count": len(items)}

    spec = route_to_spec(_route_named(app, "bulk_create"))
    assert spec is not None

    # FastAPI expects a lone, non-embedded body param as the bare value.
    assert spec.param_locations["items"] == "whole_body"
    assert spec.parameters_schema["properties"]["items"]["type"] == "array"


def test_embedded_list_body_stays_keyed():
    app = FastAPI()

    @app.post("/bulk", name="bulk_create")
    def bulk_create(items: list[str] = Body(embed=True)) -> dict:  # noqa: B008
        return {"count": len(items)}

    spec = route_to_spec(_route_named(app, "bulk_create"))
    assert spec is not None
    assert spec.param_locations["items"] == "body"


def test_embedded_model_body_is_not_flattened():
    app = FastAPI()

    class Item(BaseModel):
        name: str

    @app.post("/items", name="create_item")
    def create_item(item: Item = Body(embed=True)) -> dict:  # noqa: B008
        return {}

    spec = route_to_spec(_route_named(app, "create_item"))
    assert spec is not None
    assert spec.param_locations == {"item": "body"}
    assert "item" in spec.parameters_schema["required"]


def test_optional_model_body_is_sent_whole():
    app = FastAPI()

    class Item(BaseModel):
        name: str

    @app.post("/items", name="upsert_item")
    def upsert_item(item: Item | None = None) -> dict:
        return {}

    spec = route_to_spec(_route_named(app, "upsert_item"))
    assert spec is not None
    assert spec.param_locations == {"item": "whole_body"}


def test_path_body_name_collision_renames_body_field():
    app = FastAPI()

    class OrderPayload(BaseModel):
        id: int
        note: str

    @app.put("/orders/{id}", name="update_order")
    def update_order(id: int, payload: OrderPayload) -> dict:
        return {}

    spec = route_to_spec(_route_named(app, "update_order"))
    assert spec is not None

    assert spec.param_locations["id"] == "path"
    assert "id_body" in spec.param_locations
    assert spec.param_locations["id_body"] == "body"


def test_form_route_excluded_returns_none():
    from fastapi import Form

    app = FastAPI()

    @app.post("/login", name="login")
    def login(username: str = Form(...), password: str = Form(...)) -> dict:
        return {}

    assert route_to_spec(_route_named(app, "login")) is None


def test_description_uses_decorator_then_docstring():
    from fastapi_ai_router.decorator import ai_route

    app = FastAPI()

    @app.post("/x", name="x_a")
    @ai_route(description="Decorator description.")
    def x_a() -> None:
        """Docstring description."""
        return None

    spec_a = route_to_spec(_route_named(app, "x_a"))
    assert spec_a is not None
    assert spec_a.description == "Decorator description."

    @app.post("/y", name="y_b")
    def y_b() -> None:
        """Docstring description."""
        return None

    spec_b = route_to_spec(_route_named(app, "y_b"))
    assert spec_b is not None
    assert spec_b.description == "Docstring description."


def test_body_model_enum_defs_hoisted_to_root():
    class Color(StrEnum):
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


def test_query_enum_defs_hoisted_to_root():
    class Size(StrEnum):
        s = "s"
        m = "m"

    app = FastAPI()

    @app.get("/shirts", name="list_shirts")
    def list_shirts(size: Size | None = None) -> dict:
        return {}

    spec = route_to_spec(_route_named(app, "list_shirts"))
    assert spec is not None
    schema = spec.parameters_schema
    assert "$defs" not in schema["properties"]["size"]
    assert schema["$defs"]["Size"]["enum"] == ["s", "m"]


def test_lone_list_of_models_body_defs_hoisted_to_root():
    class Item(BaseModel):
        name: str

    app = FastAPI()

    @app.post("/items", name="bulk_items")
    def bulk_items(items: list[Item]) -> dict:
        return {}

    spec = route_to_spec(_route_named(app, "bulk_items"))
    assert spec is not None
    schema = spec.parameters_schema
    assert schema["properties"]["items"] == {"items": {"$ref": "#/$defs/Item"}, "type": "array"}
    assert "name" in schema["$defs"]["Item"]["properties"]


def test_conflicting_def_names_are_renamed_at_root():
    def _mode_a() -> type[StrEnum]:
        class Mode(StrEnum):
            a = "a"

        return Mode

    def _mode_b() -> type[StrEnum]:
        class Mode(StrEnum):
            b = "b"

        return Mode

    mode_a, mode_b = _mode_a(), _mode_b()
    app = FastAPI()

    @app.get("/x", name="pick")
    def pick(first: mode_a | None = None, second: mode_b | None = None) -> dict:  # type: ignore[valid-type]
        return {}

    spec = route_to_spec(_route_named(app, "pick"))
    assert spec is not None
    schema = spec.parameters_schema
    # Same def name, different schema: the second is renamed so both refs resolve from the root.
    assert schema["$defs"]["Mode"]["enum"] == ["a"]
    assert schema["$defs"]["Mode__second"]["enum"] == ["b"]
    assert "$defs" not in schema["properties"]["second"]
    assert {"$ref": "#/$defs/Mode__second"} in schema["properties"]["second"]["anyOf"]


def test_flattened_model_defs_win_the_root_over_a_same_named_query_def():
    def _mode_a() -> type[StrEnum]:
        class Mode(StrEnum):
            a = "a"

        return Mode

    def _mode_b() -> type[StrEnum]:
        class Mode(StrEnum):
            b = "b"

        return Mode

    mode_a, mode_b = _mode_a(), _mode_b()

    class Payload(BaseModel):
        mode: mode_b  # type: ignore[valid-type]

    app = FastAPI()

    @app.post("/y", name="both")
    def both(payload: Payload, flag: mode_a | None = None) -> dict:  # type: ignore[valid-type]
        return {}

    spec = route_to_spec(_route_named(app, "both"))
    assert spec is not None
    schema = spec.parameters_schema
    # The flattened field refs the root, so the model's Mode must own it.
    assert schema["properties"]["mode"] == {"$ref": "#/$defs/Mode"}
    assert schema["$defs"]["Mode"]["enum"] == ["b"]
    assert schema["$defs"]["Mode__flag"]["enum"] == ["a"]
    assert {"$ref": "#/$defs/Mode__flag"} in schema["properties"]["flag"]["anyOf"]


def test_def_rename_never_overwrites_an_existing_def():
    from fastapi_ai_router.schema import _hoist_defs

    defs: dict = {"Mode": {"enum": ["a"]}, "Mode__second": {"enum": ["z"]}}
    body = _hoist_defs({"$defs": {"Mode": {"enum": ["b"]}}, "$ref": "#/$defs/Mode"}, defs, "second")
    assert defs["Mode__second"] == {"enum": ["z"]}
    assert defs["Mode__second_2"] == {"enum": ["b"]}
    assert body == {"$ref": "#/$defs/Mode__second_2"}
