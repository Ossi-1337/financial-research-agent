from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from financial_research_agent.llm import (
    AnthropicProvider,
    ChatMessage,
    ChatRequest,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    StreamEventType,
    ToolCall,
    ToolDefinition,
)
from financial_research_agent.settings import ProviderSettings


def test_anthropic_chat_maps_system_structured_output_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert payload["system"] == "Stay source-bound."
        assert payload["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Analyze"}]}
        ]
        assert payload["output_config"]["format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"answer":"ok"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="system", content="Stay source-bound."),
                        ChatMessage(role="user", content="Analyze"),
                    ),
                    response_format=ResponseFormat(
                        format_type=ResponseFormatType.JSON_SCHEMA,
                        name="answer",
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    ),
                )
            )

    response = asyncio.run(scenario())
    assert response.structured_output == {"answer": "ok"}
    assert response.usage.total_tokens == 10


def test_anthropic_maps_assistant_tool_calls_and_tool_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert payload["tools"][0]["name"] == "calculate_ratio"
        assistant = payload["messages"][1]
        assert assistant["content"][1]["type"] == "tool_use"
        tool_result = payload["messages"][2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_1"
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_2",
                        "name": "calculate_ratio",
                        "input": {"numerator": 5, "denominator": 2},
                    }
                ],
                "usage": {},
            },
        )

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="user", content="Calculate"),
                        ChatMessage(
                            role="assistant",
                            content="Calling tool",
                            tool_calls=(
                                ToolCall(
                                    id="call_1",
                                    name="calculate_ratio",
                                    arguments={"numerator": 4, "denominator": 2},
                                ),
                            ),
                        ),
                        ChatMessage(
                            role="tool",
                            content='{"status":"succeeded","value":2}',
                            name="calculate_ratio",
                            tool_call_id="call_1",
                        ),
                    ),
                    tools=(
                        ToolDefinition(
                            name="calculate_ratio",
                            description="Calculate a ratio.",
                            input_schema={"type": "object"},
                        ),
                    ),
                )
            )

    response = asyncio.run(scenario())
    assert response.tool_calls[0].id == "call_2"
    assert response.tool_calls[0].arguments["numerator"] == 5


def test_anthropic_streams_text_and_tool_call() -> None:
    body = "\n\n".join(
        f"event: message\ndata: {json.dumps(event)}"
        for event in (
            {
                "type": "message_start",
                "message": {
                    "model": "claude-test",
                    "usage": {"input_tokens": 4, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Checking "},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":"Novo"}'},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        )
    )

    async def scenario():
        async with _provider(lambda _request: httpx.Response(200, content=body)) as provider:
            return tuple(
                [
                    event
                    async for event in provider.stream_chat(
                        ChatRequest(messages=(ChatMessage(role="user", content="Check"),))
                    )
                ]
            )

    events = asyncio.run(scenario())
    assert events[0].delta == "Checking "
    assert any(event.event_type == StreamEventType.TOOL_CALL for event in events)
    assert events[-1].response is not None
    assert events[-1].response.tool_calls[0].arguments == {"query": "Novo"}
    assert events[-1].response.usage.total_tokens == 6


def test_anthropic_health_and_missing_key_are_safe() -> None:
    async def healthy():
        async with _provider(
            lambda _request: httpx.Response(200, json={"data": [{"id": "claude-test"}]})
        ) as provider:
            return await provider.check_health()

    async def missing():
        async with _provider(
            lambda _request: pytest.fail("network called"),
            api_key=None,
        ) as provider:
            return await provider.check_health()

    health = asyncio.run(healthy())
    missing_health = asyncio.run(missing())
    assert health.available_models == ("claude-test",)
    assert missing_health.status == "missing_api_key"
    assert missing_health.authenticated is False


def test_anthropic_rejects_malformed_stream_json() -> None:
    async def scenario():
        async with _provider(
            lambda _request: httpx.Response(200, content="data: {not-json}\n\n")
        ) as provider:
            return tuple(
                [
                    event
                    async for event in provider.stream_chat(
                        ChatRequest(messages=(ChatMessage(role="user", content="Stream"),))
                    )
                ]
            )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.code == ProviderErrorCode.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderErrorCode.AUTHENTICATION_FAILED),
        (429, ProviderErrorCode.RATE_LIMITED),
        (504, ProviderErrorCode.TIMEOUT),
        (500, ProviderErrorCode.PROVIDER_UNAVAILABLE),
    ],
)
def test_anthropic_maps_http_errors(status: int, expected: ProviderErrorCode) -> None:
    async def scenario():
        async with _provider(
            lambda _request: httpx.Response(
                status,
                json={"error": {"type": "api_error", "message": "provider failure"}},
            )
        ) as provider:
            await provider.chat(ChatRequest(messages=(ChatMessage(role="user", content="Hello"),)))

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.code == expected
    assert "test-key" not in exc_info.value.message


class _ProviderContext:
    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        api_key: str | None = "test-key",
    ) -> None:
        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.test/v1/",
            transport=httpx.MockTransport(handler),
        )
        self.api_key = api_key

    async def __aenter__(self) -> AnthropicProvider:
        return AnthropicProvider(
            model="claude-test",
            api_key=self.api_key,
            base_url="https://api.anthropic.test/v1",
            client=self.client,
        )

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.client.aclose()


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = "test-key",
) -> _ProviderContext:
    return _ProviderContext(handler, api_key=api_key)


def _request_json(request: httpx.Request) -> dict[str, object]:
    payload = json.loads(request.content.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_anthropic_settings_factory_uses_vendor_configuration() -> None:
    provider = AnthropicProvider.from_settings(
        ProviderSettings(
            llm_provider="anthropic",
            llm_model="claude-configured",
            anthropic_api_key="secret",
            anthropic_base_url="https://anthropic.example/v1",
        )
    )
    assert provider.model == "claude-configured"
    assert provider.base_url == "https://anthropic.example/v1/"
    assert provider.metadata.metadata["api_key_configured"] == "true"
