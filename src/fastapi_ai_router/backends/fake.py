"""FakeLLMBackend — a deterministic, network-free backend for tests.

Pass `returns=` a ToolCall, None, or an Exception:
- ToolCall → returned verbatim
- None     → simulates "model didn't pick a tool"
- Exception → raised on call
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi_ai_router.backends import LLMBackend, Message, ToolCall, ToolDef


@dataclass(frozen=True)
class FakeCall:
    messages: list[Message]
    tools: list[ToolDef]


FakeReturn = ToolCall | None | BaseException


@dataclass
class FakeLLMBackend(LLMBackend):
    returns: FakeReturn
    calls: list[FakeCall] = field(default_factory=list)

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDef],
    ) -> ToolCall | None:
        self.calls.append(FakeCall(messages=list(messages), tools=list(tools)))
        if isinstance(self.returns, BaseException):
            raise self.returns
        return self.returns
