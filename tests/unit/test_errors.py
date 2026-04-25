from fastapi_ai_router.errors import (
    AIRouterError,
    DispatchError,
    LLMBackendError,
    NoRouteMatched,
    ToolSchemaTooLarge,
    UnknownTool,
)


def test_all_errors_subclass_base():
    for cls in (NoRouteMatched, UnknownTool, LLMBackendError, DispatchError, ToolSchemaTooLarge):
        assert issubclass(cls, AIRouterError)


def test_no_route_matched_carries_llm_text():
    err = NoRouteMatched(llm_text="I don't know which endpoint to call.")
    assert err.llm_text == "I don't know which endpoint to call."
    assert "I don't know" in str(err)


def test_unknown_tool_carries_tool_name():
    err = UnknownTool(tool_name="cancel_orderr")
    assert err.tool_name == "cancel_orderr"


def test_llm_backend_error_wraps_upstream():
    upstream = TimeoutError("LLM timed out")
    err = LLMBackendError("call failed", upstream=upstream)
    assert err.upstream is upstream


def test_tool_schema_too_large_carries_metrics():
    err = ToolSchemaTooLarge(tool_count=200, approx_tokens=120_000)
    assert err.tool_count == 200
    assert err.approx_tokens == 120_000


def test_dispatch_error_basic():
    err = DispatchError("transport failure")
    assert isinstance(err, AIRouterError)
