"""fastapi-ai-router — turn FastAPI routes into a natural-language-callable surface."""

from fastapi_ai_router.backends import (
    FunctionDef,
    LLMBackend,
    Message,
    ToolCall,
    ToolDef,
)
from fastapi_ai_router.core import (
    DEFAULT_FORWARD_HEADERS,
    DEFAULT_SYSTEM_PROMPT,
    AIRouter,
)
from fastapi_ai_router.decorator import AIRouteMeta, ai_route
from fastapi_ai_router.errors import (
    AIRouterError,
    DispatchError,
    LLMBackendError,
    MissingPathParams,
    NoRouteMatched,
    ToolSchemaTooLarge,
    UnknownTool,
)
from fastapi_ai_router.observability import (
    Decision,
    DecisionHook,
    ErrorEvent,
    ErrorHook,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FORWARD_HEADERS",
    "DEFAULT_SYSTEM_PROMPT",
    "AIRouteMeta",
    "AIRouter",
    "AIRouterError",
    "Decision",
    "DecisionHook",
    "DispatchError",
    "ErrorEvent",
    "ErrorHook",
    "FunctionDef",
    "LLMBackend",
    "LLMBackendError",
    "Message",
    "MissingPathParams",
    "NoRouteMatched",
    "ToolCall",
    "ToolDef",
    "ToolSchemaTooLarge",
    "UnknownTool",
    "__version__",
    "ai_route",
]
