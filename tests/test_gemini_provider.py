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
    GeminiProvider,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    StreamEventType,
    ToolCall,
    ToolDefinition,
)


def test_gemini_chat_maps_system_tools_schema_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert request.url.path == "/v1beta/models/gemini-test:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        assert payload["systemInstruction"]["parts"][0]["text"] == "Stay source-bound."
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert payload["generationConfig"]["responseSchema"]["required"] == ["answer"]
        return _response(text='{"answer":"ok"}')

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="system", content="Stay source-bound."),
                        ChatMessage(role="user", content="Analyze"),
                    ),
                    tools=(
                        ToolDefinition(
                            name="lookup",
                            description="Lookup evidence.",
                            input_schema={"type": "object"},
                        ),
                    ),
                    response_format=ResponseFormat(
                        format_type=ResponseFormatType.JSON_SCHEMA,
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                )
            )

    response = asyncio.run(scenario())
    assert response.structured_output == {"answer": "ok"}
    assert response.usage.total_tokens == 9


def test_gemini_maps_tool_call_and_tool_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        function_response = payload["contents"][2]["parts"][0]["functionResponse"]
        assert function_response["id"] == "call_1"
        assert function_response["name"] == "lookup"
        return _response(
            parts=[
                {
                    "functionCall": {
                        "id": "call_2",
                        "name": "lookup",
                        "args": {"query": "Novo"},
                    }
                }
            ],
            finish_reason="STOP",
        )

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="user", content="Lookup"),
                        ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=(
                                ToolCall(id="call_1", name="lookup", arguments={"query": "NVO"}),
                            ),
                        ),
                        ChatMessage(
                            role="tool",
                            content='{"status":"succeeded"}',
                            name="lookup",
                            tool_call_id="call_1",
                        ),
                    )
                )
            )

    response = asyncio.run(scenario())
    assert response.tool_calls[0].id == "call_2"
    assert response.finish_reason == FinishReason.TOOL_CALLS


def test_gemini_preserves_thought_signature_across_tool_rounds() -> None:
    signature = "signed-model-thought"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assistant_part = payload["contents"][1]["parts"][0]
        assert assistant_part["thoughtSignature"] == signature
        return _response(text="done")

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.chat(
                ChatRequest(
                    messages=(
                        ChatMessage(role="user", content="Lookup"),
                        ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=(
                                ToolCall(
                                    id="call_1",
                                    name="lookup",
                                    arguments={"query": "NVO"},
                                    metadata={"gemini_thought_signature": signature},
                                ),
                            ),
                        ),
                        ChatMessage(
                            role="tool",
                            content='{"status":"succeeded"}',
                            name="lookup",
                            tool_call_id="call_1",
                        ),
                    )
                )
            )

    assert asyncio.run(scenario()).message.content == "done"

    parsed = _response(
        parts=[
            {
                "thoughtSignature": signature,
                "functionCall": {"id": "call_2", "name": "lookup", "args": {}},
            }
        ]
    )

    async def parse_response():
        async with _provider(lambda _request: parsed) as provider:
            return await provider.chat(
                ChatRequest(messages=(ChatMessage(role="user", content="Lookup"),))
            )

    assert (
        asyncio.run(parse_response()).tool_calls[0].metadata["gemini_thought_signature"]
        == signature
    )


def test_gemini_batch_embeddings_preserve_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        assert request.url.path.endswith(":batchEmbedContents")
        assert [item["content"]["parts"][0]["text"] for item in payload["requests"]] == [
            "Novo",
            "Nordisk",
        ]
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]},
        )

    async def scenario():
        async with _provider(handler) as provider:
            return await provider.embed(EmbeddingRequest(input_texts=("Novo", "Nordisk")))

    response = asyncio.run(scenario())
    assert response.embeddings == ((0.1, 0.2), (0.3, 0.4))


def test_gemini_streams_text_and_reports_safety_filter() -> None:
    chunks = [
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello "}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
        },
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "world"}]},
                    "finishReason": "SAFETY",
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2},
        },
    ]
    body = "\n\n".join(f"data: {json.dumps(chunk)}" for chunk in chunks)

    async def scenario():
        async with _provider(lambda _request: httpx.Response(200, content=body)) as provider:
            return tuple(
                [
                    event
                    async for event in provider.stream_chat(
                        ChatRequest(messages=(ChatMessage(role="user", content="Stream"),))
                    )
                ]
            )

    events = asyncio.run(scenario())
    assert [event.delta for event in events if event.delta] == ["Hello ", "world"]
    assert events[-1].event_type == StreamEventType.COMPLETED
    assert events[-1].response is not None
    assert events[-1].response.finish_reason == FinishReason.CONTENT_FILTER


def test_gemini_blocked_prompt_is_content_filtered_not_malformed() -> None:
    async def scenario():
        async with _provider(
            lambda _request: httpx.Response(
                200,
                json={"promptFeedback": {"blockReason": "SAFETY"}},
            )
        ) as provider:
            return await provider.chat(
                ChatRequest(messages=(ChatMessage(role="user", content="Blocked"),))
            )

    assert asyncio.run(scenario()).finish_reason == FinishReason.CONTENT_FILTER


def test_gemini_rejects_malformed_stream_json() -> None:
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


def test_gemini_health_and_errors() -> None:
    async def healthy():
        async with _provider(
            lambda _request: httpx.Response(
                200,
                json={"models": [{"name": "models/gemini-test"}]},
            )
        ) as provider:
            return await provider.check_health()

    assert asyncio.run(healthy()).available_models == ("gemini-test",)

    async def failing():
        async with _provider(
            lambda _request: httpx.Response(
                429,
                json={"error": {"message": "rate limit"}},
            )
        ) as provider:
            await provider.chat(ChatRequest(messages=(ChatMessage(role="user", content="Hello"),)))

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(failing())
    assert exc_info.value.code == ProviderErrorCode.RATE_LIMITED


class _ProviderContext:
    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        api_key: str | None = "test-key",
    ) -> None:
        self.client = httpx.AsyncClient(
            base_url="https://generativelanguage.test/v1beta/",
            transport=httpx.MockTransport(handler),
        )
        self.api_key = api_key

    async def __aenter__(self) -> GeminiProvider:
        return GeminiProvider(
            model="gemini-test",
            api_key=self.api_key,
            base_url="https://generativelanguage.test/v1beta",
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


def _response(
    *,
    text: str | None = None,
    parts: list[dict[str, object]] | None = None,
    finish_reason: str = "STOP",
) -> httpx.Response:
    response_parts = parts if parts is not None else [{"text": text or ""}]
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": response_parts},
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 5},
        },
    )


def _request_json(request: httpx.Request) -> dict[str, object]:
    payload = json.loads(request.content.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload
