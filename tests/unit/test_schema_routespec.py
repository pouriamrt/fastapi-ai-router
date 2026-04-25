import dataclasses

import pytest

from fastapi_ai_router.schema import RouteSpec


def _sample(handler=lambda: None) -> RouteSpec:
    return RouteSpec(
        name="cancel_order",
        description="Cancel an order.",
        method="POST",
        path_template="/orders/{order_id}/cancel",
        parameters_schema={
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
            "required": ["order_id"],
        },
        param_locations={"order_id": "path"},
        handler=handler,
    )


def test_routespec_is_frozen():
    spec = _sample()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_routespec_supports_equality_by_value():
    a = _sample(handler=lambda: None)
    b = _sample(handler=a.handler)
    assert a == b
