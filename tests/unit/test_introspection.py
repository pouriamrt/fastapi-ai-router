"""Tests for mode filtering and the registry/tool-list build step."""

from fastapi import FastAPI

from fastapi_ai_router.decorator import ai_route
from fastapi_ai_router.introspection import (
    ModeConfig,
    build_registry,
    build_tools,
)


def _app_with_mixed_routes() -> FastAPI:
    app = FastAPI()

    @app.get("/public", name="public_decorated", tags=["ai"])
    @ai_route(description="Public decorated.")
    def public_decorated() -> dict:
        return {}

    @app.get("/tagged_only", name="tagged_only", tags=["ai"])
    def tagged_only() -> dict:
        return {}

    @app.get("/plain", name="plain")
    def plain() -> dict:
        return {}

    @app.get("/admin/secret", name="admin_secret")
    def admin_secret() -> dict:
        return {}

    @app.get("/dangerous", name="dangerous")
    @ai_route(expose=False)
    def dangerous() -> dict:
        return {}

    @app.get("/internal", name="internal", include_in_schema=False)
    def internal() -> dict:
        return {}

    return app


def test_mode_decorator_picks_only_decorated_with_expose_true():
    app = _app_with_mixed_routes()
    cfg = ModeConfig(mode="decorator")
    reg = build_registry(app, cfg)
    assert set(reg.keys()) == {"public_decorated"}


def test_mode_tag_picks_routes_with_matching_tag_minus_kill_switch():
    app = _app_with_mixed_routes()
    cfg = ModeConfig(mode="tag", tag="ai")
    reg = build_registry(app, cfg)
    assert set(reg.keys()) == {"public_decorated", "tagged_only"}


def test_mode_all_excludes_kill_switch_internal_and_excluded_paths():
    app = _app_with_mixed_routes()
    cfg = ModeConfig(mode="all", exclude=("/admin/*",))
    reg = build_registry(app, cfg)
    assert "admin_secret" not in reg
    assert "dangerous" not in reg
    assert "internal" not in reg
    assert {"public_decorated", "tagged_only", "plain"} <= set(reg.keys())


def test_build_tools_returns_openai_compatible_shape():
    app = _app_with_mixed_routes()
    cfg = ModeConfig(mode="decorator")
    reg = build_registry(app, cfg)
    tools = build_tools(reg)
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "public_decorated"
    assert "description" in t["function"]
    assert "parameters" in t["function"]


def test_invalid_mode_raises_at_construction():
    import pytest

    with pytest.raises(ValueError, match="mode"):
        ModeConfig(mode="bogus")  # type: ignore[arg-type]
