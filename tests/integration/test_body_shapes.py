"""Every single-body-param shape must reach the route with the value the LLM picked.

FastAPI sends a lone body param as its bare value unless it's `Body(embed=True)`,
so the loopback has to mirror that wire shape per route.
"""

from typing import Any

import pytest
from fastapi import Body, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fastapi_ai_router import AIRouter, ai_route
from fastapi_ai_router.backends import ToolCall
from fastapi_ai_router.backends.fake import FakeLLMBackend


class Item(BaseModel):
    name: str


def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/lone-list/{i}")
    @ai_route(description="lone list body")
    def lone_list(i: int, tags: list[str]) -> dict[str, Any]:
        return {"got": tags}

    @app.post("/embedded-list/{i}")
    @ai_route(description="embedded list body")
    def embedded_list(i: int, tags: list[str] = Body(embed=True)) -> dict[str, Any]:  # noqa: B008
        return {"got": tags}

    @app.post("/lone-dict")
    @ai_route(description="lone dict body")
    def lone_dict(payload: dict[str, Any] = Body()) -> dict[str, Any]:  # noqa: B008
        return {"got": payload}

    @app.post("/optional-model")
    @ai_route(description="optional model body")
    def optional_model(item: Item | None = None) -> dict[str, Any]:
        return {"got": item.model_dump() if item else None}

    @app.post("/embedded-model")
    @ai_route(description="embedded model body")
    def embedded_model(item: Item = Body(embed=True)) -> dict[str, Any]:  # noqa: B008
        return {"got": item.model_dump()}

    @app.post("/plain-model")
    @ai_route(description="plain model body")
    def plain_model(item: Item) -> dict[str, Any]:
        return {"got": item.model_dump()}

    return app


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("lone_list", {"i": 1, "tags": ["x", "y"]}, ["x", "y"]),
        ("embedded_list", {"i": 1, "tags": ["x", "y"]}, ["x", "y"]),
        ("lone_dict", {"payload": {"k": 1}}, {"k": 1}),
        ("optional_model", {"item": {"name": "n"}}, {"name": "n"}),
        ("embedded_model", {"item": {"name": "n"}}, {"name": "n"}),
        ("plain_model", {"name": "n"}, {"name": "n"}),
    ],
)
def test_route_receives_the_chosen_body(tool, args, expected):
    app = _app()
    call = ToolCall(
        name=tool, args=args, reasoning=None, prompt_tokens=0, completion_tokens=0, model="fake"
    )
    AIRouter(app, llm=FakeLLMBackend(returns=call))
    resp = TestClient(app).post("/ai", json={"query": "q"})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["result"] == {"got": expected}


@pytest.mark.parametrize(("args", "expected"), [({}, None), ({"tags": []}, [])])
def test_optional_whole_body_keeps_empty_and_omitted_values(args, expected):
    app = FastAPI()

    @app.post("/tags")
    @ai_route(description="optional lone list body")
    def set_tags(tags: list[str] | None = None) -> dict[str, Any]:
        return {"got": tags}

    call = ToolCall(
        name="set_tags", args=args, reasoning=None, prompt_tokens=0, completion_tokens=0, model="f"
    )
    AIRouter(app, llm=FakeLLMBackend(returns=call))
    resp = TestClient(app).post("/ai", json={"query": "q"})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["result"] == {"got": expected}
