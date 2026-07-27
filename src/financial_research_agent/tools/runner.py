from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from financial_research_agent.llm import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ResponseFormat,
)
from financial_research_agent.tools.contracts import (
    ToolContext,
    ToolErrorCode,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
)


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    messages: tuple[ChatMessage, ...]
    final_response: ChatResponse | None
    tool_results: tuple[ToolResult, ...]
    rounds: int
    stopped_reason: str


@dataclass(frozen=True, slots=True)
class ToolCallingRunner:
    provider: ChatProvider
    registry: ToolRegistry
    max_tool_rounds: int = 3

    def __post_init__(self) -> None:
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")

    async def run(
        self,
        messages: Iterable[ChatMessage],
        *,
        model: str | None = None,
        context: ToolContext | None = None,
        response_format: ResponseFormat | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ToolLoopResult:
        tool_context = context or ToolContext()
        history = tuple(messages)
        if not history:
            raise ValueError("messages must contain at least one message")
        tool_results: list[ToolResult] = []
        tools = self.registry.tool_definitions(tool_context)
        final_response: ChatResponse | None = None

        for round_number in range(1, self.max_tool_rounds + 1):
            response = await self.provider.chat(
                ChatRequest(
                    messages=history,
                    model=model,
                    tools=tools,
                    response_format=response_format or ResponseFormat(),
                    metadata=metadata or {},
                )
            )
            final_response = response
            if not response.tool_calls:
                history = (*history, response.message)
                return ToolLoopResult(
                    messages=history,
                    final_response=response,
                    tool_results=tuple(tool_results),
                    rounds=round_number,
                    stopped_reason="final_response",
                )

            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.message.content,
                tool_calls=response.tool_calls,
            )
            history = (*history, assistant_message)

            for tool_call in response.tool_calls:
                result = await self.registry.execute(tool_call, tool_context)
                tool_results.append(result)
                history = (
                    *history,
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=result.to_message_content(),
                        name=result.tool_name,
                        tool_call_id=result.tool_call_id,
                    ),
                )
                if result.status != ToolResultStatus.SUCCEEDED:
                    return ToolLoopResult(
                        messages=history,
                        final_response=response,
                        tool_results=tuple(tool_results),
                        rounds=round_number,
                        stopped_reason=result.error_code.value
                        if result.error_code is not None
                        else result.status.value,
                    )

        max_rounds_result = ToolResult.failed(
            tool_call_id="tool-loop:max-rounds",
            tool_name="tool_loop",
            error_code=ToolErrorCode.MAX_ROUNDS_EXCEEDED,
            errors=(f"Maximum tool rounds exceeded: {self.max_tool_rounds}",),
        )
        tool_results.append(max_rounds_result)
        return ToolLoopResult(
            messages=history,
            final_response=final_response,
            tool_results=tuple(tool_results),
            rounds=self.max_tool_rounds,
            stopped_reason=ToolErrorCode.MAX_ROUNDS_EXCEEDED.value,
        )
