from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from financial_research_agent.llm import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    EmbeddingProvider,
    EmbeddingRequest,
    FinishReason,
    MessageRole,
    OfflineTestProvider,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    ToolDefinition,
)
from financial_research_agent.llm.registry import ProviderRegistry, create_offline_provider_registry


def test_offline_provider_supports_chat_and_token_usage() -> None:
    provider = OfflineTestProvider()
    request = ChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Give me a short summary.")]
    )

    response = asyncio.run(provider.chat(request))

    assert isinstance(provider, ChatProvider)
    assert response.provider == "offline-test"
    assert response.model == "offline-test"
    assert response.message.content == "offline-test response: Give me a short summary."
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_offline_provider_can_request_tool_calls() -> None:
    provider = OfflineTestProvider()
    request = ChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Find Novo Nordisk.")],
        tools=[
            ToolDefinition(
                name="company_lookup",
                description="Resolve a company name.",
                input_schema={"type": "object"},
            )
        ],
    )

    response = asyncio.run(provider.chat(request))

    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "company_lookup"
    assert dict(response.tool_calls[0].arguments) == {}


def test_offline_provider_can_return_structured_output() -> None:
    provider = OfflineTestProvider()
    request = ChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Return JSON.")],
        response_format=ResponseFormat(
            format_type=ResponseFormatType.JSON_SCHEMA,
            name="ResearchPlan",
            json_schema={"type": "object"},
        ),
    )

    response = asyncio.run(provider.chat(request))

    assert response.structured_output is not None
    assert dict(response.structured_output)["provider"] == "offline-test"
    assert dict(response.structured_output)["schema_name"] == "ResearchPlan"


def test_offline_provider_streams_events_without_network_access() -> None:
    provider = OfflineTestProvider()
    request = ChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Stream this response.")]
    )

    events = asyncio.run(_collect(provider.stream_chat(request)))

    assert events[-1].event_type == StreamEventType.COMPLETED
    assert any(event.event_type == StreamEventType.MESSAGE_DELTA for event in events)
    assert events[-1].response is not None
    assert events[-1].response.provider == "offline-test"


def test_offline_provider_returns_deterministic_embeddings() -> None:
    provider = OfflineTestProvider()
    request = EmbeddingRequest(input_texts=["Novo Nordisk", "Novo Nordisk"])

    response = asyncio.run(provider.embed(request))

    assert isinstance(provider, EmbeddingProvider)
    assert response.provider == "offline-test"
    assert response.model == "offline-test-embedding"
    assert len(response.embeddings) == 2
    assert len(response.embeddings[0]) == 8
    assert response.embeddings[0] == response.embeddings[1]


def test_offline_provider_can_simulate_errors() -> None:
    error = ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message="Offline test failure.",
        provider="offline-test",
        retryable=True,
    )
    provider = OfflineTestProvider(fail_with=error)
    request = ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="Hello")])

    with pytest.raises(ProviderError, match="Offline test failure"):
        asyncio.run(provider.chat(request))

    events = asyncio.run(_collect(provider.stream_chat(request)))

    assert events == (StreamEvent(event_type=StreamEventType.ERROR, error=error),)


def test_provider_registry_swaps_implementations_by_name() -> None:
    registry = ProviderRegistry()
    registry.register_chat_provider("offline-a", OfflineTestProvider(provider="offline-a"))
    registry.register_chat_provider("offline-b", OfflineTestProvider(provider="offline-b"))

    request = ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="Hello")])
    response_a = asyncio.run(registry.chat_provider("offline-a").chat(request))
    response_b = asyncio.run(registry.chat_provider("offline-b").chat(request))

    assert response_a.provider == "offline-a"
    assert response_b.provider == "offline-b"
    assert registry.has_chat_provider("offline-a")
    assert not registry.has_embedding_provider("offline-a")

    with pytest.raises(ProviderError, match="not registered"):
        registry.chat_provider("missing-provider")


def test_default_registry_contains_offline_test_provider() -> None:
    registry = create_offline_provider_registry()

    assert registry.has_chat_provider("offline-test")
    assert registry.has_embedding_provider("offline-test")


async def _collect(events: AsyncIterator[StreamEvent]) -> tuple[StreamEvent, ...]:
    return tuple([event async for event in events])
