import dataclasses

import pytest

from fastapi_ai_router.decorator import AI_ROUTE_ATTR, AIRouteMeta, ai_route


def test_decorator_attaches_metadata_with_defaults():
    @ai_route()
    def cancel_order(order_id: int) -> None: ...

    meta = getattr(cancel_order, AI_ROUTE_ATTR)
    assert isinstance(meta, AIRouteMeta)
    assert meta.expose is True
    assert meta.description is None


def test_decorator_records_description():
    @ai_route(description="Cancel a customer's order.")
    def cancel_order(order_id: int) -> None: ...

    meta = getattr(cancel_order, AI_ROUTE_ATTR)
    assert meta.description == "Cancel a customer's order."


def test_expose_false_kill_switch():
    @ai_route(expose=False)
    def secret_admin_action() -> None: ...

    meta = getattr(secret_admin_action, AI_ROUTE_ATTR)
    assert meta.expose is False


def test_decorator_does_not_change_call_behavior():
    @ai_route(description="Sum two numbers.")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_meta_is_immutable():
    @ai_route(description="x")
    def fn() -> None: ...

    meta = getattr(fn, AI_ROUTE_ATTR)
    assert dataclasses.is_dataclass(meta)
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.expose = False  # type: ignore[misc]
