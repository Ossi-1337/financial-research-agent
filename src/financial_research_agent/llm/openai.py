from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

import financial_research_agent.llm.openai_compatible as _wire
from financial_research_agent.llm.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from financial_research_agent.settings import ProviderSettings

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True, slots=True)
class OpenAIModelProfile:
    model: str
    capabilities: tuple[ProviderCapability, ...]
    context_window: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class OnlineProviderHealth:
    provider: str
    base_url: str
    model: str
    reachable: bool
    authenticated: bool
    status: str
    available_models: tuple[str, ...] = ()
    capabilities: tuple[ProviderCapability, ...] = ()
    limitations: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "reachable": self.reachable,
            "authenticated": self.authenticated,
            "status": self.status,
            "available_models": list(self.available_models),
            "capabilities": [capability.value for capability in self.capabilities],
            "limitations": list(self.limitations),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_OPENAI_BASE_URL
    organization: str | None = None
    project: str | None = None
    embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    timeout_seconds: float = 30.0
    provider: str = "openai"
    client: httpx.AsyncClient | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _wire.require_text("model", self.model))
        object.__setattr__(self, "api_key", _wire.optional_text(self.api_key))
        object.__setattr__(self, "base_url", _wire.normalize_base_url(self.base_url))
        object.__setattr__(self, "organization", _wire.optional_text(self.organization))
        object.__setattr__(self, "project", _wire.optional_text(self.project))
        object.__setattr__(
            self,
            "embedding_model",
            _wire.require_text("embedding_model", self.embedding_model),
        )
        object.__setattr__(self, "provider", _wire.require_text("provider", self.provider))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_settings(cls, settings: ProviderSettings) -> Self:
        return cls(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization,
            project=settings.openai_project,
            embedding_model=settings.embedding_model or DEFAULT_OPENAI_EMBEDDING_MODEL,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    @property
    def metadata(self) -> ModelMetadata:
        profile = openai_model_profile(self.model)
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=profile.capabilities,
            context_window=profile.context_window,
            max_output_tokens=profile.max_output_tokens,
            metadata={
                "base_url": self.base_url,
                "api_key_configured": str(self.api_key is not None).lower(),
                "embedding_model": self.embedding_model,
                "organization_configured": str(self.organization is not None).lower(),
                "project_configured": str(self.project is not None).lower(),
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        selected_model = request.model or self.model
        payload = _hosted_chat_payload(request, selected_model, stream=False)
        data = await self._request_json("POST", "chat/completions", json_payload=payload)
        return _wire.chat_response_from_payload(
            data,
            self.provider,
            selected_model,
            request,
            "OpenAI API",
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        selected_model = request.model or self.model
        payload = _hosted_chat_payload(request, selected_model, stream=True)
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

        response = ChatResponse(
            message=ChatMessage(role="assistant", content="".join(content_parts)),
            provider=self.provider,
            model=selected_model,
            usage=usage,
        )
        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        selected_model = request.model or self.embedding_model
        data = await self._request_json(
            "POST",
            "embeddings",
            json_payload={"model": selected_model, "input": list(request.input_texts)},
        )
        return _wire.embedding_response_from_payload(
            data,
            self.provider,
            selected_model,
            "OpenAI API",
        )

    async def check_health(self) -> OnlineProviderHealth:
        if self.api_key is None:
            return OnlineProviderHealth(
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                reachable=False,
                authenticated=False,
                status="missing_api_key",
                capabilities=(),
                limitations=_openai_limitations(),
                error="OpenAI API key is not configured.",
            )

        try:
            available_models = await self.list_models()
        except ProviderError as exc:
            authenticated = exc.code != ProviderErrorCode.AUTHENTICATION_FAILED
            reachable = exc.code not in (
                ProviderErrorCode.PROVIDER_UNAVAILABLE,
                ProviderErrorCode.TIMEOUT,
            )
            return OnlineProviderHealth(
                provider=self.provider,
                base_url=self.base_url,
                model=self.model,
                reachable=reachable,
                authenticated=authenticated,
                status="unreachable" if not reachable else "error",
                capabilities=(),
                limitations=_openai_limitations(),
                error=exc.message,
            )

        profile = openai_model_profile(self.model)
        return OnlineProviderHealth(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            reachable=True,
            authenticated=True,
            status="ok",
            available_models=available_models,
            capabilities=profile.capabilities,
            limitations=_openai_limitations(),
        )

    async def list_models(self) -> tuple[str, ...]:
        data = await self._request_json("GET", "models")
        return _wire.model_ids_from_models_payload(data)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        response = await self._request(method, path, json_payload=json_payload)
        if response.status_code >= 400:
            _wire.raise_http_error(response, self.provider, self.model, "OpenAI API")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"OpenAI API returned malformed JSON from {path}.",
                provider=self.provider,
                model=self.model,
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"OpenAI API returned non-object JSON from {path}.",
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
                service_label="OpenAI API",
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
                    "OpenAI API",
                ) from exc
            except httpx.RequestError as exc:
                raise _wire.provider_error_from_request_error(
                    exc,
                    self.provider,
                    self.model,
                    "OpenAI API",
                ) from exc

    def _client_context(self) -> _wire.AsyncClientContext:
        headers = self._headers()
        if self.client is not None:
            self.client.headers.update(headers)
            return _wire.AsyncClientContext(self.client, close=False)
        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers=headers,
        )
        return _wire.AsyncClientContext(client, close=True)

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            raise ProviderError(
                code=ProviderErrorCode.AUTHENTICATION_FAILED,
                message=(
                    "OpenAI API key is not configured. Set FRA_OPENAI_API_KEY or OPENAI_API_KEY."
                ),
                provider=self.provider,
                model=self.model,
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "financial-research-agent/openai",
        }
        if self.organization is not None:
            headers["OpenAI-Organization"] = self.organization
        if self.project is not None:
            headers["OpenAI-Project"] = self.project
        return headers


def openai_model_profile(model: str) -> OpenAIModelProfile:
    normalized = _wire.require_text("model", model).lower()
    if normalized.startswith("gpt-5.5"):
        return OpenAIModelProfile(
            model=model,
            capabilities=_chat_capabilities(),
            context_window=1_050_000,
            max_output_tokens=128_000,
        )
    if normalized.startswith("gpt-5.4-mini"):
        return OpenAIModelProfile(
            model=model,
            capabilities=_chat_capabilities(),
            context_window=400_000,
            max_output_tokens=128_000,
        )
    if normalized.startswith("gpt-5.4"):
        return OpenAIModelProfile(
            model=model,
            capabilities=_chat_capabilities(),
            context_window=1_050_000,
            max_output_tokens=128_000,
        )
    if normalized.startswith("gpt-5") or normalized == "chat-latest":
        return OpenAIModelProfile(
            model=model,
            capabilities=_chat_capabilities(),
            context_window=400_000,
            max_output_tokens=128_000,
        )
    if normalized.startswith("text-embedding-"):
        return OpenAIModelProfile(
            model=model,
            capabilities=(ProviderCapability.EMBEDDINGS, ProviderCapability.TOKEN_ACCOUNTING),
            context_window=8192,
        )
    return OpenAIModelProfile(model=model, capabilities=_wire.default_capabilities())


def _hosted_chat_payload(request: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    payload = _wire.chat_request_payload(request, model, stream=stream)
    max_tokens = payload.pop("max_tokens", None)
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    return payload


def _chat_capabilities() -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability.CHAT,
        ProviderCapability.STREAMING,
        ProviderCapability.TOKEN_ACCOUNTING,
        ProviderCapability.TOOL_CALLS,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def _openai_limitations() -> tuple[str, ...]:
    return (
        "Requests may incur hosted provider cost when explicit OpenAI credentials are configured.",
        "This adapter uses Chat Completions for parity with local OpenAI-compatible runtimes.",
        "Some current OpenAI model features are exposed through the Responses API "
        "and are deferred.",
        "Rate limits, model access, and regional settings depend on the configured OpenAI account.",
    )
