from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

import httpx

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
    ProviderErrorCode,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from financial_research_agent.settings import ProviderSettings

DEFAULT_LOCAL_OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"


class LocalRuntime(StrEnum):
    LLAMA_CPP = "llama.cpp"
    OLLAMA = "ollama"
    GENERIC_OPENAI = "generic-openai"


@dataclass(frozen=True, slots=True)
class LocalEndpointHealth:
    provider: str
    runtime: LocalRuntime
    base_url: str
    model: str
    reachable: bool
    status: str
    available_models: tuple[str, ...] = ()
    capabilities: tuple[ProviderCapability, ...] = ()
    limitations: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "runtime": self.runtime.value,
            "base_url": self.base_url,
            "model": self.model,
            "reachable": self.reachable,
            "status": self.status,
            "available_models": list(self.available_models),
            "capabilities": [capability.value for capability in self.capabilities],
            "limitations": list(self.limitations),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class OpenAICompatibleLocalProvider:
    model: str
    base_url: str = DEFAULT_LOCAL_OPENAI_BASE_URL
    runtime: LocalRuntime = LocalRuntime.LLAMA_CPP
    timeout_seconds: float = 30.0
    provider: str = "local-openai"
    client: httpx.AsyncClient | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _require_text("model", self.model))
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        object.__setattr__(self, "runtime", LocalRuntime(self.runtime))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_settings(cls, settings: ProviderSettings) -> Self:
        return cls(
            model=settings.llm_model,
            base_url=settings.llm_base_url or DEFAULT_LOCAL_OPENAI_BASE_URL,
            runtime=LocalRuntime(settings.llm_local_runtime),
            timeout_seconds=settings.llm_timeout_seconds,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=_default_capabilities(),
            metadata={
                "base_url": self.base_url,
                "runtime": self.runtime.value,
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = _chat_request_payload(request, request.model or self.model, stream=False)
        data = await self._request_json("POST", "chat/completions", json_payload=payload)
        return _chat_response_from_payload(
            data,
            self.provider,
            request.model or self.model,
            request,
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = _chat_request_payload(request, request.model or self.model, stream=True)
        content_parts: list[str] = []
        usage = TokenUsage()

        async for data in self._stream_json("POST", "chat/completions", json_payload=payload):
            choices = data.get("choices", [])
            if not choices:
                usage = _usage_from_payload(data.get("usage"))
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content_delta = delta.get("content")
            if isinstance(content_delta, str) and content_delta:
                content_parts.append(content_delta)
                yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=content_delta)
            try:
                tool_calls = _tool_calls_from_payload(delta)
            except ProviderError:
                tool_calls = ()
            for tool_call in tool_calls:
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL, tool_call=tool_call)

        content = "".join(content_parts)
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            provider=self.provider,
            model=request.model or self.model,
            usage=usage,
        )
        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        selected_model = request.model or self.model
        data = await self._request_json(
            "POST",
            "embeddings",
            json_payload={"model": selected_model, "input": list(request.input_texts)},
        )
        return _embedding_response_from_payload(data, self.provider, selected_model)

    async def check_health(self) -> LocalEndpointHealth:
        available_models: tuple[str, ...] = ()
        health_status = "unknown"
        error: str | None = None

        try:
            health_status = await self._health_status()
            available_models = await self._available_models()
        except ProviderError as exc:
            error = exc.message
            if exc.code == ProviderErrorCode.INVALID_REQUEST and "health" in exc.message:
                try:
                    available_models = await self._available_models()
                except ProviderError as model_error:
                    return LocalEndpointHealth(
                        provider=self.provider,
                        runtime=self.runtime,
                        base_url=self.base_url,
                        model=self.model,
                        reachable=False,
                        status="unreachable",
                        capabilities=(),
                        limitations=_limitations_for_runtime(self.runtime),
                        error=model_error.message,
                    )
                health_status = "models_available"
                error = None
            else:
                return LocalEndpointHealth(
                    provider=self.provider,
                    runtime=self.runtime,
                    base_url=self.base_url,
                    model=self.model,
                    reachable=False,
                    status="unreachable",
                    capabilities=(),
                    limitations=_limitations_for_runtime(self.runtime),
                    error=error,
                )

        return LocalEndpointHealth(
            provider=self.provider,
            runtime=self.runtime,
            base_url=self.base_url,
            model=self.model,
            reachable=True,
            status=health_status,
            available_models=available_models,
            capabilities=_default_capabilities(),
            limitations=_limitations_for_runtime(self.runtime),
            error=error,
        )

    async def _health_status(self) -> str:
        response = await self._request("GET", "health")
        if response.status_code == 404:
            raise ProviderError(
                code=ProviderErrorCode.INVALID_REQUEST,
                message="Local endpoint does not expose health.",
                provider=self.provider,
                model=self.model,
            )
        if response.status_code >= 400:
            _raise_http_error(response, self.provider, self.model)
        try:
            data = response.json()
        except ValueError:
            return "ok"
        status = data.get("status")
        return status if isinstance(status, str) else "ok"

    async def _available_models(self) -> tuple[str, ...]:
        data = await self._request_json("GET", "models")
        values = data.get("data", [])
        if isinstance(values, list):
            models = tuple(
                model_id
                for item in values
                if isinstance(item, Mapping)
                for model_id in (_model_id_from_payload(item),)
                if model_id is not None
            )
            if models:
                return models
        return ()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        response = await self._request(method, path, json_payload=json_payload)
        if response.status_code >= 400:
            _raise_http_error(response, self.provider, self.model)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Local endpoint returned malformed JSON from {path}.",
                provider=self.provider,
                model=self.model,
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Local endpoint returned non-object JSON from {path}.",
                provider=self.provider,
                model=self.model,
            )
        return data

    async def _stream_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any],
    ) -> AsyncIterator[Mapping[str, Any]]:
        async with self._client_context() as client:
            try:
                async with client.stream(method, path, json=json_payload) as response:
                    if response.status_code >= 400:
                        _raise_http_error(response, self.provider, self.model)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line.removeprefix("data:").strip()
                        if payload == "[DONE]":
                            return
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(
                                code=ProviderErrorCode.MALFORMED_RESPONSE,
                                message=f"Local endpoint returned malformed SSE JSON from {path}.",
                                provider=self.provider,
                                model=self.model,
                            ) from exc
                        if not isinstance(data, Mapping):
                            raise ProviderError(
                                code=ProviderErrorCode.MALFORMED_RESPONSE,
                                message=f"Local endpoint returned non-object SSE JSON from {path}.",
                                provider=self.provider,
                                model=self.model,
                            )
                        yield data
            except httpx.TimeoutException as exc:
                raise _provider_error_from_timeout(exc, self.provider, self.model) from exc
            except httpx.RequestError as exc:
                raise _provider_error_from_request_error(exc, self.provider, self.model) from exc

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        async with self._client_context() as client:
            try:
                return await client.request(method, path, json=json_payload)
            except httpx.TimeoutException as exc:
                raise _provider_error_from_timeout(exc, self.provider, self.model) from exc
            except httpx.RequestError as exc:
                raise _provider_error_from_request_error(exc, self.provider, self.model) from exc

    def _client_context(self) -> _AsyncClientContext:
        if self.client is not None:
            return _AsyncClientContext(self.client, close=False)
        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers={"User-Agent": "financial-research-agent/local-openai"},
        )
        return _AsyncClientContext(client, close=True)


@dataclass(slots=True)
class _AsyncClientContext:
    client: httpx.AsyncClient
    close: bool

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, *_exc_info: object) -> None:
        if self.close:
            await self.client.aclose()


def _chat_request_payload(request: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_payload(message) for message in request.messages],
        "stream": stream,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": _mutable_json_value(tool.input_schema),
                },
            }
            for tool in request.tools
        ]
    if request.response_format.format_type == ResponseFormatType.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    elif request.response_format.format_type == ResponseFormatType.JSON_SCHEMA:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_format.name or "structured_output",
                "schema": _mutable_json_value(request.response_format.json_schema or {}),
            },
        }
    return payload


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _chat_response_from_payload(
    data: Mapping[str, Any],
    provider: str,
    model: str,
    request: ChatRequest,
) -> ChatResponse:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned a chat response without choices.",
            provider=provider,
            model=model,
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned a malformed chat choice.",
            provider=provider,
            model=model,
        )
    message_payload = choice.get("message", {})
    if not isinstance(message_payload, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned a malformed chat message.",
            provider=provider,
            model=model,
        )
    content = message_payload.get("content") or ""
    if not isinstance(content, str):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned non-string chat content.",
            provider=provider,
            model=model,
        )
    tool_calls = _tool_calls_from_payload(message_payload)
    structured_output = _structured_output_from_content(content, request, provider, model)
    finish_reason = choice.get("finish_reason") or FinishReason.STOP.value

    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
        provider=provider,
        model=str(data.get("model") or model),
        finish_reason=FinishReason(finish_reason),
        tool_calls=tool_calls,
        structured_output=structured_output,
        usage=_usage_from_payload(data.get("usage")),
    )


def _structured_output_from_content(
    content: str,
    request: ChatRequest,
    provider: str,
    model: str,
) -> Mapping[str, Any] | None:
    if request.response_format.format_type == ResponseFormatType.TEXT:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned malformed structured output JSON.",
            provider=provider,
            model=model,
        ) from exc
    if not isinstance(data, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned non-object structured output JSON.",
            provider=provider,
            model=model,
        )
    return data


def _embedding_response_from_payload(
    data: Mapping[str, Any],
    provider: str,
    model: str,
) -> EmbeddingResponse:
    values = data.get("data")
    if not isinstance(values, list) or not values:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Local endpoint returned an embedding response without data.",
            provider=provider,
            model=model,
        )
    ordered = sorted(
        (item for item in values if isinstance(item, Mapping)),
        key=lambda item: int(item.get("index", 0)),
    )
    embeddings = []
    for item in ordered:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message="Local endpoint returned a malformed embedding.",
                provider=provider,
                model=model,
            )
        embeddings.append(tuple(float(value) for value in embedding))
    return EmbeddingResponse(
        embeddings=tuple(embeddings),
        provider=provider,
        model=str(data.get("model") or model),
        usage=_usage_from_payload(data.get("usage")),
    )


def _tool_calls_from_payload(payload: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    values = payload.get("tool_calls", ())
    if not isinstance(values, list):
        return ()
    tool_calls: list[ToolCall] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            continue
        function = value.get("function", {})
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or name.strip() == "":
            continue
        arguments = _tool_arguments(function.get("arguments"))
        tool_calls.append(
            ToolCall(
                id=str(value.get("id") or f"tool-call:{index}"),
                name=name,
                arguments=arguments,
            )
        )
    return tuple(tool_calls)


def _tool_arguments(value: object) -> Mapping[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message="Local endpoint returned malformed tool call arguments.",
                provider="local-openai",
            ) from exc
        if isinstance(data, Mapping):
            return data
    raise ProviderError(
        code=ProviderErrorCode.MALFORMED_RESPONSE,
        message="Local endpoint returned non-object tool call arguments.",
        provider="local-openai",
    )


def _usage_from_payload(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0),
        output_tokens=int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0),
    )


def _raise_http_error(response: httpx.Response, provider: str, model: str) -> None:
    status_code = response.status_code
    if status_code in (400, 422):
        code = ProviderErrorCode.INVALID_REQUEST
        retryable = False
    elif status_code in (401, 403):
        code = ProviderErrorCode.AUTHENTICATION_FAILED
        retryable = False
    elif status_code == 429:
        code = ProviderErrorCode.RATE_LIMITED
        retryable = True
    elif status_code >= 500:
        code = ProviderErrorCode.PROVIDER_UNAVAILABLE
        retryable = True
    else:
        code = ProviderErrorCode.PROVIDER_UNAVAILABLE
        retryable = False
    raise ProviderError(
        code=code,
        message=f"Local endpoint returned HTTP {status_code}.",
        provider=provider,
        model=model,
        retryable=retryable,
    )


def _provider_error_from_timeout(
    error: httpx.TimeoutException, provider: str, model: str
) -> ProviderError:
    return ProviderError(
        code=ProviderErrorCode.TIMEOUT,
        message=f"Local endpoint timed out: {error}",
        provider=provider,
        model=model,
        retryable=True,
    )


def _provider_error_from_request_error(
    error: httpx.RequestError, provider: str, model: str
) -> ProviderError:
    return ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message=f"Local endpoint is unavailable: {error}",
        provider=provider,
        model=model,
        retryable=True,
    )


def _model_id_from_payload(value: Mapping[str, Any]) -> str | None:
    model_id = value.get("id", value.get("model", value.get("name")))
    return model_id if isinstance(model_id, str) and model_id.strip() else None


def _limitations_for_runtime(runtime: LocalRuntime) -> tuple[str, ...]:
    common = (
        "Tool calling depends on model, chat template, and server support.",
        "Structured output depends on model behavior and endpoint support for response_format.",
        "Embeddings require a loaded embedding-capable model or compatible pooling setup.",
    )
    if runtime == LocalRuntime.LLAMA_CPP:
        return (
            *common,
            "llama.cpp JSON/schema support and tool behavior vary by build, flags, and model.",
        )
    if runtime == LocalRuntime.OLLAMA:
        return (
            *common,
            "Ollama OpenAI compatibility covers common endpoints but is not full OpenAI parity.",
        )
    return common


def _default_capabilities() -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability.CHAT,
        ProviderCapability.STREAMING,
        ProviderCapability.EMBEDDINGS,
        ProviderCapability.TOKEN_ACCOUNTING,
        ProviderCapability.TOOL_CALLS,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def _mutable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_mutable_json_value(item) for item in value]
    return value


def _normalize_base_url(value: str) -> str:
    return _require_text("base_url", value).rstrip("/") + "/"


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
