from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from financial_research_agent.llm.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)


@dataclass(frozen=True, slots=True)
class OfflineTestProvider:
    provider: str = "offline-test"
    model: str = "offline-test"
    embedding_model: str = "offline-test-embedding"
    fail_with: ProviderError | None = None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=(
                ProviderCapability.CHAT,
                ProviderCapability.TOOL_CALLS,
                ProviderCapability.STRUCTURED_OUTPUT,
                ProviderCapability.EMBEDDINGS,
                ProviderCapability.STREAMING,
                ProviderCapability.TOKEN_ACCOUNTING,
            ),
            context_window=8192,
            max_output_tokens=2048,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self._raise_if_configured()
        selected_model = request.model or self.model

        if request.response_format.format_type != ResponseFormatType.TEXT:
            payload = {
                "message_count": len(request.messages),
                "model": selected_model,
                "provider": self.provider,
            }
            if request.response_format.name is not None:
                payload["schema_name"] = request.response_format.name
            content = json.dumps(payload, sort_keys=True)
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                provider=self.provider,
                model=selected_model,
                structured_output=payload,
                usage=_usage_for(request, content),
            )

        tool_calls: tuple[ToolCall, ...] = ()
        finish_reason = FinishReason.STOP
        if request.tools:
            tool_calls = (
                ToolCall(
                    id=f"tool-call:{self.provider}:1",
                    name=request.tools[0].name,
                    arguments={},
                ),
            )
            finish_reason = FinishReason.TOOL_CALLS

        content = _offline_content(request)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            ),
            provider=self.provider,
            model=selected_model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=_usage_for(request, content),
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        try:
            response = await self.chat(request)
        except ProviderError as error:
            yield StreamEvent(event_type=StreamEventType.ERROR, error=error)
            return

        for tool_call in response.tool_calls:
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL, tool_call=tool_call)

        if response.structured_output is not None:
            yield StreamEvent(
                event_type=StreamEventType.STRUCTURED_OUTPUT,
                structured_output=response.structured_output,
            )

        for delta in response.message.content.split():
            yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=delta)

        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._raise_if_configured()
        selected_model = request.model or self.embedding_model
        embeddings = tuple(_embedding_for_text(text) for text in request.input_texts)
        return EmbeddingResponse(
            embeddings=embeddings,
            provider=self.provider,
            model=selected_model,
            usage=TokenUsage(input_tokens=sum(_token_count(text) for text in request.input_texts)),
        )

    def _raise_if_configured(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with


def _offline_content(request: ChatRequest) -> str:
    last_user_message = next(
        (
            message.content
            for message in reversed(request.messages)
            if message.role == MessageRole.USER
        ),
        request.messages[-1].content,
    )
    if last_user_message.strip() == "":
        return "offline-test response"
    return f"offline-test response: {last_user_message.strip()}"


def _usage_for(request: ChatRequest, content: str) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(_token_count(message.content) for message in request.messages),
        output_tokens=_token_count(content),
    )


def _token_count(text: str) -> int:
    return len(text.split())


def _embedding_for_text(text: str) -> tuple[float, ...]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple(round(byte / 255, 6) for byte in digest[:8])
