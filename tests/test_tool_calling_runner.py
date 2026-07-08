from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ModelMetadata,
    OfflineTestProvider,
    ToolCall,
)
from financial_research_agent.tools import (
    ToolCallingRunner,
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolResultStatus,
    create_default_tool_registry,
)


def test_tool_calling_runner_executes_tool_then_returns_final_response() -> None:
    provider = ScriptedToolProvider(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                provider="scripted",
                model="scripted",
                tool_calls=(ToolCall(id="call_time", name="current_utc_datetime", arguments={}),),
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done."),
                provider="scripted",
                model="scripted",
            ),
        ]
    )
    runner = ToolCallingRunner(provider=provider, registry=create_default_tool_registry())

    result = asyncio.run(
        runner.run(
            [ChatMessage(role=MessageRole.USER, content="What time is it?")],
            context=_tool_context("current_utc_datetime"),
        )
    )

    assert result.stopped_reason == "final_response"
    assert result.final_response is not None
    assert result.final_response.message.content == "Done."
    assert result.tool_results[0].status == ToolResultStatus.SUCCEEDED
    assert provider.requests[1].messages[-2].tool_calls[0].id == "call_time"
    tool_payload = json.loads(provider.requests[1].messages[-1].content)
    assert tool_payload["tool_call_id"] == "call_time"


def test_tool_calling_runner_executes_multiple_tool_calls_in_one_round() -> None:
    provider = ScriptedToolProvider(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                provider="scripted",
                model="scripted",
                tool_calls=(
                    ToolCall(id="call_time", name="current_utc_datetime", arguments={}),
                    ToolCall(
                        id="call_ratio",
                        name="calculate_ratio",
                        arguments={"numerator": 10, "denominator": 4, "precision": 1},
                    ),
                ),
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done."),
                provider="scripted",
                model="scripted",
            ),
        ]
    )
    runner = ToolCallingRunner(provider=provider, registry=create_default_tool_registry())

    result = asyncio.run(
        runner.run(
            [ChatMessage(role="user", content="Use tools.")],
            context=_tool_context("current_utc_datetime", "calculate_ratio"),
        )
    )

    assert [tool_result.tool_call_id for tool_result in result.tool_results] == [
        "call_time",
        "call_ratio",
    ]
    assert result.tool_results[1].data["ratio"] == "2.5"
    assert result.stopped_reason == "final_response"


def test_tool_calling_runner_stops_on_failed_tool_result() -> None:
    provider = ScriptedToolProvider(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content=""),
                provider="scripted",
                model="scripted",
                tool_calls=(
                    ToolCall(
                        id="call_bad",
                        name="calculate_ratio",
                        arguments={"numerator": 1, "denominator": 0},
                    ),
                ),
            ),
        ]
    )
    runner = ToolCallingRunner(provider=provider, registry=create_default_tool_registry())

    result = asyncio.run(
        runner.run(
            [ChatMessage(role="user", content="Divide.")],
            context=_tool_context("calculate_ratio"),
        )
    )

    assert result.stopped_reason == ToolErrorCode.DIVISION_BY_ZERO.value
    assert result.tool_results[0].status == ToolResultStatus.FAILED
    assert len(provider.requests) == 1


def test_tool_calling_runner_reports_max_rounds_exceeded() -> None:
    provider = RepeatingToolProvider()
    runner = ToolCallingRunner(
        provider=provider,
        registry=create_default_tool_registry(),
        max_tool_rounds=1,
    )

    result = asyncio.run(
        runner.run(
            [ChatMessage(role="user", content="Loop.")],
            context=_tool_context("current_utc_datetime"),
        )
    )

    assert result.stopped_reason == ToolErrorCode.MAX_ROUNDS_EXCEEDED.value
    assert result.tool_results[-1].error_code == ToolErrorCode.MAX_ROUNDS_EXCEEDED
    assert result.rounds == 1


def test_tool_calling_runner_can_execute_offline_test_provider_tool_call() -> None:
    runner = ToolCallingRunner(
        provider=OfflineTestProvider(),
        registry=create_default_tool_registry(),
        max_tool_rounds=1,
    )

    result = asyncio.run(
        runner.run(
            [ChatMessage(role="user", content="Use a tool.")],
            context=_tool_context("current_utc_datetime"),
        )
    )

    assert result.tool_results[0].tool_name == "current_utc_datetime"
    assert result.tool_results[0].status == ToolResultStatus.SUCCEEDED


@dataclass
class ScriptedToolProvider:
    responses: list[ChatResponse]

    def __post_init__(self) -> None:
        self.requests: list[ChatRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(provider="scripted", model="scripted")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    def stream_chat(self, _request: ChatRequest):
        raise NotImplementedError


@dataclass
class RepeatingToolProvider:
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(provider="scripted", model="scripted")

    async def chat(self, _request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content=""),
            provider="scripted",
            model="scripted",
            tool_calls=(ToolCall(id="call_time", name="current_utc_datetime", arguments={}),),
        )

    def stream_chat(self, _request: ChatRequest):
        raise NotImplementedError


def _tool_context(*tool_names: str) -> ToolContext:
    return ToolContext(
        allowed_permissions=tuple(ToolPermission),
        allowed_tools=tool_names,
    )
