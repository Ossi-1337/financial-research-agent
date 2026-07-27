from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

import financial_research_agent.llm.openai_compatible as _wire
from financial_research_agent.llm.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ProviderHealth,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from financial_research_agent.settings import (
    DEFAULT_ANTHROPIC_API_VERSION,
    DEFAULT_ANTHROPIC_BASE_URL,
    ProviderSettings,
)

DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 1024


def _capabilities() -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability.CHAT,
        ProviderCapability.STREAMING,
        ProviderCapability.TOKEN_ACCOUNTING,
        ProviderCapability.TOOL_CALLS,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def _limitations() -> tuple[str, ...]:
    return (
        "Anthropic does not expose an embedding API through this adapter.",
        "Sampling parameters and structured output support depend on the selected model.",
        "Requests may incur hosted provider cost when credentials are configured.",
    )


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    api_version: str = DEFAULT_ANTHROPIC_API_VERSION
    timeout_seconds: float = 30.0
    provider: str = "anthropic"
    client: httpx.AsyncClient | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _wire.require_text("model", self.model))
        object.__setattr__(self, "api_key", _wire.optional_text(self.api_key))
        object.__setattr__(self, "base_url", _wire.normalize_base_url(self.base_url))
        object.__setattr__(self, "api_version", _wire.require_text("api_version", self.api_version))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_settings(cls, settings: ProviderSettings) -> Self:
        return cls(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            api_version=settings.anthropic_api_version,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=_capabilities(),
            metadata={
                "base_url": self.base_url,
                "api_key_configured": str(self.api_key is not None).lower(),
                "api_version": self.api_version,
                "embeddings": "unsupported",
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        selected_model = request.model or self.model
        data = await self._request_json(
            "POST",
            "messages",
            json_payload=_chat_payload(request, selected_model, stream=False),
            model=selected_model,
        )
        return _chat_response(data, request, selected_model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        selected_model = request.model or self.model
        payload = _chat_payload(request, selected_model, stream=True)
        content_parts: list[str] = []
        tool_blocks: dict[int, dict[str, Any]] = {}
        usage = TokenUsage()
        response_model = selected_model
        finish_reason = FinishReason.STOP

        async for data in self._stream_events("POST", "messages", payload, selected_model):
            event_type = data.get("type")
            if event_type == "message_start":
                message = data.get("message")
                if isinstance(message, Mapping):
                    response_model = str(message.get("model") or selected_model)
                    usage = _usage(message.get("usage"))
            elif event_type == "content_block_start":
                index = int(data.get("index", 0))
                block = data.get("content_block")
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    tool_blocks[index] = {
                        "id": str(block.get("id") or f"anthropic-tool:{index}"),
                        "name": str(block.get("name") or ""),
                        "json": "",
                        "input": block.get("input"),
                    }
            elif event_type == "content_block_delta":
                index = int(data.get("index", 0))
                delta = data.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    text = str(delta["text"])
                    content_parts.append(text)
                    yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=text)
                elif delta.get("type") == "input_json_delta" and index in tool_blocks:
                    partial = delta.get("partial_json")
                    if isinstance(partial, str):
                        tool_blocks[index]["json"] += partial
            elif event_type == "message_delta":
                delta = data.get("delta")
                if isinstance(delta, Mapping):
                    finish_reason = _finish_reason(delta.get("stop_reason"))
                delta_usage = _usage(data.get("usage"))
                usage = TokenUsage(
                    input_tokens=max(usage.input_tokens, delta_usage.input_tokens),
                    output_tokens=max(usage.output_tokens, delta_usage.output_tokens),
                )

        tool_calls = tuple(_stream_tool_call(index, block) for index, block in tool_blocks.items())
        for tool_call in tool_calls:
            yield StreamEvent(event_type=StreamEventType.TOOL_CALL, tool_call=tool_call)
        content = "".join(content_parts)
        structured_output = (
            None
            if tool_calls
            else _wire.structured_output_from_content(
                content,
                request,
                self.provider,
                response_model,
                "Anthropic API",
            )
        )
        response = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            ),
            provider=self.provider,
            model=response_model,
            finish_reason=FinishReason.TOOL_CALLS if tool_calls else finish_reason,
            tool_calls=tool_calls,
            structured_output=structured_output,
            usage=usage,
        )
        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)

    async def check_health(self) -> ProviderHealth:
        if self.api_key is None:
            return ProviderHealth(
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                reachable=False,
                authenticated=False,
                status="missing_api_key",
                limitations=_limitations(),
                error="Anthropic API key is not configured.",
            )
        try:
            models = await self.list_models()
        except ProviderError as exc:
            authenticated = exc.code != ProviderErrorCode.AUTHENTICATION_FAILED
            reachable = exc.code not in {
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                ProviderErrorCode.TIMEOUT,
            }
            return ProviderHealth(
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                reachable=reachable,
                authenticated=authenticated,
                status="unreachable" if not reachable else "error",
                limitations=_limitations(),
                error=exc.message,
            )
        return ProviderHealth(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            reachable=True,
            authenticated=True,
            status="ok",
            available_models=models,
            capabilities=_capabilities(),
            limitations=_limitations(),
        )

    async def list_models(self) -> tuple[str, ...]:
        data = await self._request_json("GET", "models", model=self.model)
        values = data.get("data")
        if not isinstance(values, list):
            return ()
        return tuple(
            str(item["id"])
            for item in values
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        model: str,
    ) -> Mapping[str, Any]:
        async with self._client_context() as client:
            try:
                response = await client.request(method, path, json=json_payload)
            except httpx.TimeoutException as exc:
                raise _wire.provider_error_from_timeout(
                    exc, self.provider, model, "Anthropic API"
                ) from exc
            except httpx.RequestError as exc:
                raise _wire.provider_error_from_request_error(
                    exc, self.provider, model, "Anthropic API"
                ) from exc
        if response.status_code >= 400:
            _wire.raise_http_error(response, self.provider, model, "Anthropic API")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Anthropic API returned malformed JSON from {path}.",
                provider=self.provider,
                model=model,
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Anthropic API returned non-object JSON from {path}.",
                provider=self.provider,
                model=model,
            )
        return data

    async def _stream_events(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any],
        model: str,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async with self._client_context() as client:
            try:
                async with client.stream(method, path, json=payload) as response:
                    if response.status_code >= 400:
                        _wire.raise_http_error(response, self.provider, model, "Anthropic API")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line.removeprefix("data:").strip()
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(
                                code=ProviderErrorCode.MALFORMED_RESPONSE,
                                message="Anthropic API returned malformed SSE JSON.",
                                provider=self.provider,
                                model=model,
                            ) from exc
                        if not isinstance(data, Mapping):
                            raise ProviderError(
                                code=ProviderErrorCode.MALFORMED_RESPONSE,
                                message="Anthropic API returned non-object SSE JSON.",
                                provider=self.provider,
                                model=model,
                            )
                        yield data
            except httpx.TimeoutException as exc:
                raise _wire.provider_error_from_timeout(
                    exc, self.provider, model, "Anthropic API"
                ) from exc
            except httpx.RequestError as exc:
                raise _wire.provider_error_from_request_error(
                    exc, self.provider, model, "Anthropic API"
                ) from exc

    def _client_context(self) -> _wire.AsyncClientContext:
        headers = self._headers()
        if self.client is not None:
            self.client.headers.update(headers)
            return _wire.AsyncClientContext(self.client, close=False)
        return _wire.AsyncClientContext(
            httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                headers=headers,
            ),
            close=True,
        )

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            raise ProviderError(
                code=ProviderErrorCode.AUTHENTICATION_FAILED,
                message=(
                    "Anthropic API key is not configured. Set FRA_ANTHROPIC_API_KEY "
                    "or ANTHROPIC_API_KEY."
                ),
                provider=self.provider,
                model=self.model,
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "User-Agent": "financial-research-agent/anthropic",
        }


def _chat_payload(request: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    system_messages = [message.content for message in request.messages if message.role == "system"]
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_output_tokens or DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
        "messages": [
            value
            for message in request.messages
            if message.role != MessageRole.SYSTEM
            for value in (_message_payload(message),)
        ],
        "stream": stream,
    }
    if system_messages:
        payload["system"] = "\n\n".join(system_messages)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": _wire.mutable_json_value(tool.input_schema),
            }
            for tool in request.tools
        ]
    if request.response_format.format_type != ResponseFormatType.TEXT:
        schema = (
            request.response_format.json_schema
            if request.response_format.format_type == ResponseFormatType.JSON_SCHEMA
            else {"type": "object", "additionalProperties": True}
        )
        payload["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _wire.mutable_json_value(schema or {}),
            }
        }
    return payload


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role == MessageRole.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": _tool_result_is_error(message.content),
                }
            ],
        }
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    if message.role == MessageRole.ASSISTANT:
        blocks.extend(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": _wire.mutable_json_value(tool_call.arguments),
            }
            for tool_call in message.tool_calls
        )
    return {
        "role": "assistant" if message.role == MessageRole.ASSISTANT else "user",
        "content": blocks or [{"type": "text", "text": ""}],
    }


def _chat_response(
    data: Mapping[str, Any],
    request: ChatRequest,
    selected_model: str,
) -> ChatResponse:
    content_blocks = data.get("content")
    if not isinstance(content_blocks, list):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Anthropic API returned a response without content blocks.",
            provider="anthropic",
            model=selected_model,
        )
    content = "".join(
        str(block.get("text"))
        for block in content_blocks
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    tool_calls = tuple(
        ToolCall(
            id=str(block.get("id") or f"anthropic-tool:{index}"),
            name=str(block.get("name")),
            arguments=block.get("input") if isinstance(block.get("input"), Mapping) else {},
        )
        for index, block in enumerate(content_blocks)
        if isinstance(block, Mapping)
        and block.get("type") == "tool_use"
        and isinstance(block.get("name"), str)
    )
    model = str(data.get("model") or selected_model)
    structured_output = (
        None
        if tool_calls
        else _wire.structured_output_from_content(
            content,
            request,
            "anthropic",
            model,
            "Anthropic API",
        )
    )
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        provider="anthropic",
        model=model,
        finish_reason=(
            FinishReason.TOOL_CALLS if tool_calls else _finish_reason(data.get("stop_reason"))
        ),
        tool_calls=tool_calls,
        structured_output=structured_output,
        usage=_usage(data.get("usage")),
    )


def _usage(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(value.get("input_tokens", 0) or 0),
        output_tokens=int(value.get("output_tokens", 0) or 0),
    )


def _finish_reason(value: object) -> FinishReason:
    return {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "model_context_window_exceeded": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
        "refusal": FinishReason.CONTENT_FILTER,
    }.get(str(value), FinishReason.STOP)


def _stream_tool_call(index: int, block: Mapping[str, Any]) -> ToolCall:
    arguments = block.get("input")
    raw = block.get("json")
    if isinstance(raw, str) and raw:
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message="Anthropic API returned malformed streamed tool arguments.",
                provider="anthropic",
            ) from exc
    if not isinstance(arguments, Mapping):
        arguments = {}
    return ToolCall(
        id=str(block.get("id") or f"anthropic-tool:{index}"),
        name=str(block.get("name") or f"tool_{index}"),
        arguments=arguments,
    )


def _tool_result_is_error(content: str) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, Mapping) and payload.get("status") not in {None, "succeeded"}
