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
    FinishReason,
    LocalRuntime,
    MessageRole,
    OpenAICompatibleLocalProvider,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolDefinition,
    create_default_provider_registry,
)
from financial_research_agent.settings import ProviderSettings


def test_local_openai_provider_parses_chat_response() -> None:
    async def scenario() -> tuple[str, int, int]:
        async with _provider(_chat_handler("Hello from local model.")) as provider:
            response = await provider.chat(
                ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="Hello")])
            )
        return response.message.content, response.usage.input_tokens, response.usage.output_tokens

    content, input_tokens, output_tokens = asyncio.run(scenario())

    assert content == "Hello from local model."
    assert input_tokens == 4
    assert output_tokens == 5


def test_local_openai_provider_sends_and_parses_tool_calls() -> None:
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

    async def scenario() -> tuple[FinishReason, str, dict[str, object]]:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[ChatMessage(role=MessageRole.USER, content="Find Novo Nordisk.")],
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
        return response.finish_reason, tool_call.name, dict(tool_call.arguments)

    finish_reason, tool_name, arguments = asyncio.run(scenario())

    assert finish_reason == FinishReason.TOOL_CALLS
    assert tool_name == "company_lookup"
    assert arguments == {"query": "Novo Nordisk"}


def test_local_openai_provider_serializes_assistant_tool_call_messages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assistant_message = payload["messages"][1]
        tool_message = payload["messages"][2]
        assert assistant_message["role"] == "assistant"
        assert assistant_message["tool_calls"][0]["id"] == "call_1"
        assert assistant_message["tool_calls"][0]["function"]["name"] == "company_lookup"
        assert json.loads(assistant_message["tool_calls"][0]["function"]["arguments"]) == {
            "query": "Novo Nordisk"
        }
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_1"
        return _chat_response(content="Done.")

    async def scenario() -> str:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[
                        ChatMessage(role=MessageRole.USER, content="Find Novo Nordisk."),
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="company_lookup",
                                    arguments={"query": "Novo Nordisk"},
                                )
                            ],
                        ),
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content='{"matches":[]}',
                            tool_call_id="call_1",
                        ),
                    ]
                )
            )
        return response.message.content

    assert asyncio.run(scenario()) == "Done."


def test_local_openai_provider_parses_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "CompanySummary"
        return _chat_response(content=json.dumps({"company": "Novo Nordisk", "ok": True}))

    async def scenario() -> dict[str, object]:
        async with _provider(handler) as provider:
            response = await provider.chat(
                ChatRequest(
                    messages=[ChatMessage(role=MessageRole.USER, content="Return JSON.")],
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


def test_local_openai_provider_parses_embeddings_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload == {"model": "local-model", "input": ["Novo", "Nordisk"]}
        return httpx.Response(
            200,
            json={
                "model": "local-model",
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


def test_local_openai_provider_parses_streaming_sse_response() -> None:
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
                        ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="Stream")])
                    )
                ]
            )

    events = asyncio.run(scenario())

    assert [event.delta for event in events if event.delta is not None] == ["Hello", " world"]
    assert events[-1].event_type == StreamEventType.COMPLETED
    assert events[-1].response is not None
    assert events[-1].response.message.content == "Hello world"


def test_local_openai_provider_reports_health_and_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        return httpx.Response(404)

    async def scenario() -> tuple[bool, bool, str, tuple[str, ...], tuple[ProviderCapability, ...]]:
        async with _provider(handler) as provider:
            health = await provider.check_health()
        return (
            health.reachable,
            health.ready,
            health.status,
            health.available_models,
            health.capabilities,
        )

    reachable, ready, status, models, capabilities = asyncio.run(scenario())

    assert reachable is True
    assert ready is True
    assert status == "ok"
    assert models == ("local-model",)
    assert ProviderCapability.CHAT in capabilities


def test_local_openai_provider_tolerates_missing_health_when_models_work() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-model"}]})
        return httpx.Response(404)

    async def scenario() -> tuple[bool, bool, str, tuple[str, ...]]:
        async with _provider(handler) as provider:
            health = await provider.check_health()
        return health.reachable, health.ready, health.status, health.available_models

    assert asyncio.run(scenario()) == (True, True, "models_available", ("local-model",))


def test_local_openai_provider_reports_llama_cpp_model_loading_as_not_ready() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(
            503,
            json={"error": {"code": 503, "message": "Loading model", "type": "server_error"}},
        )

    async def scenario() -> tuple[bool, bool, str, str | None]:
        async with _provider(handler) as provider:
            health = await provider.check_health()
        return health.reachable, health.ready, health.status, health.error

    assert asyncio.run(scenario()) == (
        True,
        False,
        "loading",
        "Local model is still loading.",
    )
    assert requests == ["/v1/health"]


def test_local_openai_provider_does_not_treat_unknown_503_as_loading() -> None:
    async def scenario() -> tuple[bool, bool, str]:
        async with _provider(
            lambda _request: httpx.Response(503, json={"error": {"message": "Unavailable"}})
        ) as provider:
            health = await provider.check_health()
        return health.reachable, health.ready, health.status

    assert asyncio.run(scenario()) == (False, False, "unreachable")


def test_local_openai_provider_requires_models_when_health_is_not_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    async def scenario() -> tuple[bool, bool, str]:
        async with _provider(handler) as provider:
            health = await provider.check_health()
        return health.reachable, health.ready, health.status

    assert asyncio.run(scenario()) == (True, False, "no_models")


@pytest.mark.parametrize(
    ("status_code", "expected_code", "retryable"),
    [
        (400, ProviderErrorCode.INVALID_REQUEST, False),
        (401, ProviderErrorCode.AUTHENTICATION_FAILED, False),
        (429, ProviderErrorCode.RATE_LIMITED, True),
        (500, ProviderErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
def test_local_openai_provider_maps_http_errors(
    status_code: int,
    expected_code: ProviderErrorCode,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        async with _provider(lambda _request: httpx.Response(status_code)) as provider:
            await provider.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hello")]))

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is retryable


def test_local_openai_provider_maps_transport_timeout_and_malformed_response() -> None:
    request = ChatRequest(messages=[ChatMessage(role="user", content="Hello")])

    async def timeout_scenario() -> None:
        def handler(http_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=http_request)

        async with _provider(handler) as provider:
            await provider.chat(request)

    async def connection_scenario() -> None:
        def handler(http_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=http_request)

        async with _provider(handler) as provider:
            await provider.chat(request)

    async def malformed_scenario() -> None:
        async with _provider(lambda _request: httpx.Response(200, content=b"not-json")) as provider:
            await provider.chat(request)

    with pytest.raises(ProviderError) as timeout_error:
        asyncio.run(timeout_scenario())
    with pytest.raises(ProviderError) as connection_error:
        asyncio.run(connection_scenario())
    with pytest.raises(ProviderError) as malformed_error:
        asyncio.run(malformed_scenario())

    assert timeout_error.value.code == ProviderErrorCode.TIMEOUT
    assert connection_error.value.code == ProviderErrorCode.PROVIDER_UNAVAILABLE
    assert malformed_error.value.code == ProviderErrorCode.MALFORMED_RESPONSE


def test_default_provider_registry_registers_local_openai_provider() -> None:
    registry = create_default_provider_registry(
        ProviderSettings(
            llm_provider="local-openai",
            llm_model="local-model",
            llm_base_url="http://127.0.0.1:8080/v1",
            llm_local_runtime=LocalRuntime.LLAMA_CPP,
        )
    )

    assert registry.has_chat_provider("offline-test")
    assert registry.has_chat_provider("local-openai")
    assert registry.has_embedding_provider("local-openai")


class _ProviderContext:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._client = httpx.AsyncClient(
            base_url="http://local.test/v1/",
            transport=httpx.MockTransport(handler),
        )

    async def __aenter__(self) -> OpenAICompatibleLocalProvider:
        return OpenAICompatibleLocalProvider(
            model="local-model",
            base_url="http://local.test/v1",
            client=self._client,
        )

    async def __aexit__(self, *_exc_info: object) -> None:
        await self._client.aclose()


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> _ProviderContext:
    return _ProviderContext(handler)


def _chat_handler(content: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "local-model"
        return _chat_response(content=content)

    return handler


def _chat_response(
    *,
    content: str,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, object]] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "local-model",
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
