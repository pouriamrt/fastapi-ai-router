def test_public_api_exports():
    from fastapi_ai_router import (
        DEFAULT_FORWARD_HEADERS,  # noqa: F401
        DEFAULT_SYSTEM_PROMPT,  # noqa: F401
        AIRouteMeta,  # noqa: F401
        AIRouter,
        AIRouterError,  # noqa: F401
        Decision,  # noqa: F401
        DecisionHook,  # noqa: F401
        DispatchError,  # noqa: F401
        ErrorEvent,  # noqa: F401
        ErrorHook,  # noqa: F401
        FunctionDef,  # noqa: F401
        LLMBackend,  # noqa: F401
        LLMBackendError,  # noqa: F401
        Message,  # noqa: F401
        NoRouteMatched,  # noqa: F401
        ToolCall,  # noqa: F401
        ToolDef,  # noqa: F401
        ToolSchemaTooLarge,  # noqa: F401
        UnknownTool,  # noqa: F401
        ai_route,
    )

    assert callable(ai_route)
    assert callable(AIRouter)
