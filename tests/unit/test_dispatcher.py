import pytest
from fastapi import FastAPI, Header, HTTPException

from fastapi_ai_router.dispatcher import dispatch, split_by_location
from fastapi_ai_router.schema import RouteSpec


def test_split_by_location_partitions_correctly():
    args = {"order_id": 1, "limit": 20, "reason": "x"}
    locations = {"order_id": "path", "limit": "query", "reason": "body"}
    path, query, body = split_by_location(args, locations)
    assert path == {"order_id": 1}
    assert query == {"limit": 20}
    assert body == {"reason": "x"}


def test_split_handles_missing_keys():
    args = {"order_id": 1}
    locations = {"order_id": "path", "reason": "body"}
    path, _query, body = split_by_location(args, locations)
    assert path == {"order_id": 1}
    assert body == {}


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.post("/orders/{order_id}/cancel")
    def cancel(order_id: int, reason: str | None = None) -> dict:
        return {"order_id": order_id, "reason": reason}

    @app.get("/echo/{slug}")
    def echo(slug: str) -> dict:
        return {"slug": slug}

    @app.get("/auth-required")
    def auth_required(authorization: str = Header(...)) -> dict:
        if authorization != "Bearer good":
            raise HTTPException(status_code=401)
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_dispatch_calls_route_via_loopback():
    app = _build_app()
    spec = RouteSpec(
        name="cancel",
        description="",
        method="POST",
        path_template="/orders/{order_id}/cancel",
        parameters_schema={"type": "object"},
        param_locations={"order_id": "path", "reason": "body"},
        handler=lambda: None,
    )
    args = {"order_id": 7, "reason": "duplicate"}
    # NOTE: FastAPI classifies `reason` as a query parameter (non-Body simple type),
    # so to verify loopback round-trip we override param_locations to match how
    # FastAPI actually expects to receive the value. The dispatcher itself is
    # location-blind — it honors whatever spec.param_locations says.
    spec_for_dispatch = RouteSpec(
        name=spec.name,
        description=spec.description,
        method=spec.method,
        path_template=spec.path_template,
        parameters_schema=spec.parameters_schema,
        param_locations={"order_id": "path", "reason": "query"},
        handler=spec.handler,
    )
    response = await dispatch(
        spec=spec_for_dispatch,
        args=args,
        app=app,
        request_headers={},
        forward=frozenset(),
    )
    assert response.status_code == 200
    assert response.json() == {"order_id": 7, "reason": "duplicate"}


@pytest.mark.asyncio
async def test_dispatch_url_encodes_path_args_with_special_chars():
    app = _build_app()
    spec = RouteSpec(
        name="echo",
        description="",
        method="GET",
        path_template="/echo/{slug}",
        parameters_schema={"type": "object"},
        param_locations={"slug": "path"},
        handler=lambda: None,
    )
    # Use special chars that exercise URL-encoding without conflicting with
    # FastAPI/Starlette path-segment semantics (encoded slashes are rejected
    # by Starlette routing in some versions). Spaces, `+`, and `:` round-trip
    # through quote() and back through FastAPI's path parser cleanly.
    raw_slug = "abc:def+more space"
    response = await dispatch(
        spec=spec,
        args={"slug": raw_slug},
        app=app,
        request_headers={},
        forward=frozenset(),
    )
    assert response.status_code == 200
    assert response.json() == {"slug": raw_slug}


@pytest.mark.asyncio
async def test_dispatch_forwards_authorization_header():
    app = _build_app()
    spec = RouteSpec(
        name="auth_required",
        description="",
        method="GET",
        path_template="/auth-required",
        parameters_schema={"type": "object"},
        param_locations={},
        handler=lambda: None,
    )
    response = await dispatch(
        spec=spec,
        args={},
        app=app,
        request_headers={"authorization": "Bearer good", "x-bogus": "drop-me"},
        forward=frozenset({"authorization"}),
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_dispatch_does_not_forward_unallowed_headers():
    app = _build_app()
    spec = RouteSpec(
        name="auth_required",
        description="",
        method="GET",
        path_template="/auth-required",
        parameters_schema={"type": "object"},
        param_locations={},
        handler=lambda: None,
    )
    response = await dispatch(
        spec=spec,
        args={},
        app=app,
        request_headers={"authorization": "Bearer good"},
        forward=frozenset(),
    )
    # The header was not forwarded, so the dispatched request reaches
    # FastAPI without `authorization`. FastAPI returns 422 for the missing
    # required header (validation error) rather than reaching the handler's
    # 401 branch — either way, the request was rejected, which is what we
    # care about. Assert non-success.
    assert response.status_code >= 400
    assert response.status_code != 200
