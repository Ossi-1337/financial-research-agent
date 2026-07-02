from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    MessageRole,
    OpenAIProvider,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    ToolDefinition,
    create_default_provider_registry,
    openai_model_profile,
)
from financial_research_agent.settings import ProviderSettings


def test_openai_provider_sends_auth_headers_and_parses_chat_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["openai-organization"] == "org_123"
        assert request.headers["openai-project"] == "proj_123"
        assert payload["model"] == "gpt-5.5"
        assert payload["max_completion_tokens"] == 100
        assert "max_tokens" not in payload
        return _chat_response(content="Hello from OpenAI.")

    async def scenario() -> tuple[str, int, int]:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
                    max_output_tokens=100,
                )
            )
        return response.message.content, response.usage.input_tokens, response.usage.output_tokens

    content, input_tokens, output_tokens = asyncio.run(scenario())

    assert content == "Hello from OpenAI."
    assert input_tokens == 4
    assert output_tokens == 5


def test_openai_provider_sends_tools_and_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload["tools"][0]["function"]["name"] == "company_lookup"
        return _chat_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "company_lookup",
                        "arguments": json.dumps({"query": "Novo Nordisk"}),
                    },
                }
            ],
        )

    async def scenario() -> tuple[str, dict[str, object]]:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[ChatMessage(role="user", content="Find Novo Nordisk.")],
                    tools=[
                        ToolDefinition(
                            name="company_lookup",
                            description="Resolve company names.",
                            input_schema={"type": "object"},
                        )
                    ],
                )
            )
        tool_call = response.tool_calls[0]
        return tool_call.name, dict(tool_call.arguments)

    assert asyncio.run(scenario()) == ("company_lookup", {"query": "Novo Nordisk"})


def test_openai_provider_sends_and_parses_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "CompanySummary"
        return _chat_response(content=json.dumps({"company": "Novo Nordisk", "ok": True}))

    async def scenario() -> dict[str, object]:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[ChatMessage(role="user", content="Return JSON.")],
                    response_format=ResponseFormat(
                        format_type=ResponseFormatType.JSON_SCHEMA,
                        name="CompanySummary",
                        json_schema={"type": "object"},
                    ),
                )
            )
        assert response.structured_output is not None
        return dict(response.structured_output)

    assert asyncio.run(scenario()) == {"company": "Novo Nordisk", "ok": True}


def test_openai_provider_parses_embeddings_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload == {"model": "text-embedding-3-small", "input": ["Novo", "Nordisk"]}
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 2},
            },
        )

    async def scenario() -> tuple[tuple[float, ...], ...]:
        async with _provider(handler) as provider:
            response = await provider.embed(EmbeddingRequest(input_texts=["Novo", "Nordisk"]))
        return response.embeddings

    assert asyncio.run(scenario()) == ((0.1, 0.2), (0.3, 0.4))


def test_openai_provider_parses_streaming_sse_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload["stream"] is True
        body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, content=body)

    async def scenario() -> tuple[StreamEvent, ...]:
        async with _provider(handler) as provider:
            return tuple(
                [
                    event
                    async for event in provider.stream_chat(
                        ChatRequest(messages=[ChatMessage(role="user", content="Stream")])
                    )
                ]
            )

    events = asyncio.run(scenario())

    assert [event.delta for event in events if event.delta is not None] == ["Hello", " world"]
    assert events[-1].event_type == StreamEventType.COMPLETED
    assert events[-1].response is not None
    assert events[-1].response.message.content == "Hello world"


def test_openai_provider_reports_health_from_models_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "gpt-5.5"}]})

    async def scenario() -> tuple[bool, bool, str, tuple[str, ...]]:
        async with _provider(handler) as provider:
            health = await provider.check_health()
        return health.reachable, health.authenticated, health.status, health.available_models

    assert asyncio.run(scenario()) == (True, True, "ok", ("gpt-5.5",))


def test_openai_provider_missing_api_key_fails_without_network() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"data": []})

    async def health_scenario() -> tuple[bool, bool, str]:
        async with _provider(handler, api_key=None) as provider:
            health = await provider.check_health()
        return health.reachable, health.authenticated, health.status

    async def chat_scenario() -> None:
        async with _provider(handler, api_key=None) as provider:
            await provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hello")]))

    assert asyncio.run(health_scenario()) == (False, False, "missing_api_key")
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(chat_scenario())
    assert exc_info.value.code == ProviderErrorCode.AUTHENTICATION_FAILED
    assert called is False


@pytest.mark.parametrize(
    ("status_code", "body", "expected_code", "retryable"),
    [
        (401, {"error": {"message": "bad key"}}, ProviderErrorCode.AUTHENTICATION_FAILED, False),
        (429, {"error": {"message": "slow down"}}, ProviderErrorCode.RATE_LIMITED, True),
        (
            400,
            {"error": {"message": "context length exceeded", "code": "context_length_exceeded"}},
            ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED,
            False,
        ),
        (500, {"error": {"message": "server error"}}, ProviderErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
def test_openai_provider_maps_http_errors(
    status_code: int,
    body: dict[str, object],
    expected_code: ProviderErrorCode,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        async with _provider(lambda _request: httpx.Response(status_code, json=body)) as provider:
            await provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hello")]))

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is retryable


def test_default_provider_registry_registers_openai_provider() -> None:
    registry = create_default_provider_registry(
        ProviderSettings(
            llm_provider="openai",
            llm_model="gpt-5.5",
            openai_api_key="test-key",
        )
    )

    assert registry.has_chat_provider("offline-test")
    assert registry.has_chat_provider("local-openai")
    assert registry.has_chat_provider("openai")
    assert registry.has_embedding_provider("openai")


def test_openai_model_profile_exposes_capability_and_context_map() -> None:
    profile = openai_model_profile("gpt-5.5")
    embedding_profile = openai_model_profile("text-embedding-3-small")

    assert ProviderCapability.TOOL_CALLS in profile.capabilities
    assert ProviderCapability.STRUCTURED_OUTPUT in profile.capabilities
    assert profile.context_window == 1_050_000
    assert profile.max_output_tokens == 128_000
    assert embedding_profile.capabilities == (
        ProviderCapability.EMBEDDINGS,
        ProviderCapability.TOKEN_ACCOUNTING,
    )
    assert embedding_profile.context_window == 8192


class _ProviderContext:
    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        api_key: str | None = "test-key",
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.test/v1/",
            transport=httpx.MockTransport(handler),
        )
        self._api_key = api_key

    async def __aenter__(self) -> OpenAIProvider:
        return OpenAIProvider(
            model="gpt-5.5",
            api_key=self._api_key,
            base_url="https://api.openai.test/v1",
            organization="org_123",
            project="proj_123",
            client=self._client,
        )

    async def __aexit__(self, *_exc_info: object) -> None:
        await self._client.aclose()


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "test-key",
) -> _ProviderContext:
    return _ProviderContext(handler, api_key=api_key)


def _chat_response(
    *,
    content: str,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, object]] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-5.5",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or [],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5},
        },
    )


def _request_json(request: httpx.Request) -> dict[str, object]:
    value = json.loads(request.content.decode("utf-8"))
    assert isinstance(value, dict)
    return value
