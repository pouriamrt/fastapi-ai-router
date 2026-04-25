"""Tests for OpenAPI → RouteSpec projection."""

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


def test_body_as_list_is_wrapped():
    app = FastAPI()

    @app.post("/bulk", name="bulk_create")
    def bulk_create(items: list[str] = Body(...)) -> dict:  # noqa: B008
        return {"count": len(items)}

    spec = route_to_spec(_route_named(app, "bulk_create"))
    assert spec is not None

    assert "items" in spec.param_locations
    assert spec.param_locations["items"] == "body"
    assert spec.parameters_schema["properties"]["items"]["type"] == "array"


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
