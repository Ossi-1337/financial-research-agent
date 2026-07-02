from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

import httpx

import financial_research_agent.llm.openai_compatible as _wire
from financial_research_agent.llm.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    StreamEvent,
    StreamEventType,
    TokenUsage,
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
        object.__setattr__(self, "model", _wire.require_text("model", self.model))
        object.__setattr__(self, "base_url", _wire.normalize_base_url(self.base_url))
        object.__setattr__(self, "runtime", LocalRuntime(self.runtime))
        object.__setattr__(self, "provider", _wire.require_text("provider", self.provider))
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
            capabilities=_wire.default_capabilities(),
            metadata={
                "base_url": self.base_url,
                "runtime": self.runtime.value,
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = _wire.chat_request_payload(request, request.model or self.model, stream=False)
        data = await self._request_json("POST", "chat/completions", json_payload=payload)
        return _wire.chat_response_from_payload(
            data,
            self.provider,
            request.model or self.model,
            request,
            "Local endpoint",
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = _wire.chat_request_payload(request, request.model or self.model, stream=True)
        content_parts: list[str] = []
        usage = TokenUsage()

        async for data in self._stream_json("POST", "chat/completions", json_payload=payload):
            choices = data.get("choices", [])
            if not choices:
                usage = _wire.usage_from_payload(data.get("usage"))
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content_delta = delta.get("content")
            if isinstance(content_delta, str) and content_delta:
                content_parts.append(content_delta)
                yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=content_delta)
            try:
                tool_calls = _wire.tool_calls_from_payload(delta, self.provider)
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
        return _wire.embedding_response_from_payload(
            data,
            self.provider,
            selected_model,
            "Local endpoint",
        )

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
            capabilities=_wire.default_capabilities(),
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
            _wire.raise_http_error(response, self.provider, self.model, "Local endpoint")
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
            models = _wire.model_ids_from_models_payload({"data": values})
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
            _wire.raise_http_error(response, self.provider, self.model, "Local endpoint")
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
            async for data in _wire.stream_json_lines(
                client,
                method,
                path,
                json_payload=json_payload,
                provider=self.provider,
                model=self.model,
                service_label="Local endpoint",
            ):
                yield data

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
                raise _wire.provider_error_from_timeout(
                    exc,
                    self.provider,
                    self.model,
                    "Local endpoint",
                ) from exc
            except httpx.RequestError as exc:
                raise _wire.provider_error_from_request_error(
                    exc,
                    self.provider,
                    self.model,
                    "Local endpoint",
                ) from exc

    def _client_context(self) -> _wire.AsyncClientContext:
        if self.client is not None:
            return _wire.AsyncClientContext(self.client, close=False)
        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers={"User-Agent": "financial-research-agent/local-openai"},
        )
        return _wire.AsyncClientContext(client, close=True)


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
