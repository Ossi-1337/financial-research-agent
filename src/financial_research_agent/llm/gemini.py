from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self
from urllib.parse import quote

import httpx

import financial_research_agent.llm.openai_compatible as _wire
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
    ProviderHealth,
    ResponseFormatType,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
)
from financial_research_agent.settings import DEFAULT_GEMINI_BASE_URL, ProviderSettings

DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


def _capabilities() -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability.CHAT,
        ProviderCapability.STREAMING,
        ProviderCapability.EMBEDDINGS,
        ProviderCapability.TOKEN_ACCOUNTING,
        ProviderCapability.TOOL_CALLS,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def _limitations() -> tuple[str, ...]:
    return (
        "Model capabilities and safety behavior depend on the selected Gemini model.",
        "This adapter uses stateless generateContent so conversation state remains app-owned.",
        "Requests may incur hosted provider cost when credentials are configured.",
    )


@dataclass(frozen=True, slots=True)
class GeminiProvider:
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_GEMINI_BASE_URL
    embedding_model: str = DEFAULT_GEMINI_EMBEDDING_MODEL
    timeout_seconds: float = 30.0
    provider: str = "gemini"
    client: httpx.AsyncClient | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _wire.require_text("model", self.model))
        object.__setattr__(self, "api_key", _wire.optional_text(self.api_key))
        object.__setattr__(self, "base_url", _wire.normalize_base_url(self.base_url))
        object.__setattr__(
            self,
            "embedding_model",
            _wire.require_text("embedding_model", self.embedding_model),
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_settings(cls, settings: ProviderSettings) -> Self:
        return cls(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            embedding_model=settings.embedding_model or DEFAULT_GEMINI_EMBEDDING_MODEL,
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
                "embedding_model": self.embedding_model,
                "conversation_state": "app_owned",
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        selected_model = request.model or self.model
        data = await self._request_json(
            "POST",
            _model_path(selected_model, "generateContent"),
            json_payload=_chat_payload(request),
            model=selected_model,
        )
        return _chat_response(data, request, selected_model)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        selected_model = request.model or self.model
        path = f"{_model_path(selected_model, 'streamGenerateContent')}?alt=sse"
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = TokenUsage()
        finish_reason = FinishReason.STOP

        async with self._client_context() as client:
            async for data in _wire.stream_json_lines(
                client,
                "POST",
                path,
                json_payload=_chat_payload(request),
                provider=self.provider,
                model=selected_model,
                service_label="Gemini API",
            ):
                parsed = _candidate_parts(data, selected_model)
                for text in parsed.text_parts:
                    content_parts.append(text)
                    yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=text)
                for tool_call in parsed.tool_calls:
                    if tool_call.id not in {existing.id for existing in tool_calls}:
                        tool_calls.append(tool_call)
                        yield StreamEvent(
                            event_type=StreamEventType.TOOL_CALL,
                            tool_call=tool_call,
                        )
                usage = _usage(data.get("usageMetadata"))
                finish_reason = parsed.finish_reason

        content = "".join(content_parts)
        structured_output = (
            None
            if tool_calls
            else _wire.structured_output_from_content(
                content,
                request,
                self.provider,
                selected_model,
                "Gemini API",
            )
        )
        response = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tuple(tool_calls),
            ),
            provider=self.provider,
            model=selected_model,
            finish_reason=FinishReason.TOOL_CALLS if tool_calls else finish_reason,
            tool_calls=tuple(tool_calls),
            structured_output=structured_output,
            usage=usage,
        )
        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        selected_model = request.model or self.embedding_model
        model_name = _model_name(selected_model)
        data = await self._request_json(
            "POST",
            f"models/{quote(model_name, safe='-._')}:batchEmbedContents",
            json_payload={
                "requests": [
                    {
                        "model": f"models/{model_name}",
                        "content": {"parts": [{"text": text}]},
                    }
                    for text in request.input_texts
                ]
            },
            model=selected_model,
        )
        values = data.get("embeddings")
        if not isinstance(values, list) or len(values) != len(request.input_texts):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message="Gemini API returned an unexpected embedding count.",
                provider=self.provider,
                model=selected_model,
            )
        embeddings: list[tuple[float, ...]] = []
        for value in values:
            if not isinstance(value, Mapping) or not isinstance(value.get("values"), list):
                raise ProviderError(
                    code=ProviderErrorCode.MALFORMED_RESPONSE,
                    message="Gemini API returned a malformed embedding.",
                    provider=self.provider,
                    model=selected_model,
                )
            embeddings.append(tuple(float(item) for item in value["values"]))
        return EmbeddingResponse(
            embeddings=tuple(embeddings),
            provider=self.provider,
            model=selected_model,
        )

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
                error="Gemini API key is not configured.",
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
        values = data.get("models")
        if not isinstance(values, list):
            return ()
        return tuple(
            _model_name(str(item["name"]))
            for item in values
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
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
                    exc, self.provider, model, "Gemini API"
                ) from exc
            except httpx.RequestError as exc:
                raise _wire.provider_error_from_request_error(
                    exc, self.provider, model, "Gemini API"
                ) from exc
        if response.status_code >= 400:
            _wire.raise_http_error(response, self.provider, model, "Gemini API")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Gemini API returned malformed JSON from {path}.",
                provider=self.provider,
                model=model,
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderError(
                code=ProviderErrorCode.MALFORMED_RESPONSE,
                message=f"Gemini API returned non-object JSON from {path}.",
                provider=self.provider,
                model=model,
            )
        return data

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
                    "Gemini API key is not configured. Set FRA_GEMINI_API_KEY or GEMINI_API_KEY."
                ),
                provider=self.provider,
                model=self.model,
            )
        return {
            "x-goog-api-key": self.api_key,
            "User-Agent": "financial-research-agent/gemini",
        }


@dataclass(frozen=True, slots=True)
class _CandidateParts:
    text_parts: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason


def _chat_payload(request: ChatRequest) -> dict[str, Any]:
    system_messages = [message.content for message in request.messages if message.role == "system"]
    payload: dict[str, Any] = {
        "contents": [
            _message_payload(message)
            for message in request.messages
            if message.role != MessageRole.SYSTEM
        ]
    }
    if system_messages:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_messages)}]}
    if request.tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _wire.mutable_json_value(tool.input_schema),
                    }
                    for tool in request.tools
                ]
            }
        ]
    generation_config: dict[str, Any] = {}
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_output_tokens
    if request.response_format.format_type != ResponseFormatType.TEXT:
        generation_config["responseMimeType"] = "application/json"
        if request.response_format.format_type == ResponseFormatType.JSON_SCHEMA:
            generation_config["responseSchema"] = _wire.mutable_json_value(
                request.response_format.json_schema or {}
            )
    if generation_config:
        payload["generationConfig"] = generation_config
    return payload


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role == MessageRole.TOOL:
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": message.tool_call_id,
                        "name": message.name or "tool",
                        "response": _tool_response(message.content),
                    }
                }
            ],
        }
    parts: list[dict[str, Any]] = []
    if message.content:
        parts.append({"text": message.content})
    if message.role == MessageRole.ASSISTANT:
        parts.extend(
            {
                "functionCall": {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": _wire.mutable_json_value(tool_call.arguments),
                }
            }
            for tool_call in message.tool_calls
        )
    return {
        "role": "model" if message.role == MessageRole.ASSISTANT else "user",
        "parts": parts or [{"text": ""}],
    }


def _chat_response(
    data: Mapping[str, Any],
    request: ChatRequest,
    selected_model: str,
) -> ChatResponse:
    parsed = _candidate_parts(data, selected_model)
    content = "".join(parsed.text_parts)
    structured_output = (
        None
        if parsed.tool_calls
        else _wire.structured_output_from_content(
            content,
            request,
            "gemini",
            selected_model,
            "Gemini API",
        )
    )
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=parsed.tool_calls,
        ),
        provider="gemini",
        model=selected_model,
        finish_reason=FinishReason.TOOL_CALLS if parsed.tool_calls else parsed.finish_reason,
        tool_calls=parsed.tool_calls,
        structured_output=structured_output,
        usage=_usage(data.get("usageMetadata")),
    )


def _candidate_parts(data: Mapping[str, Any], model: str) -> _CandidateParts:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, Mapping) and prompt_feedback.get("blockReason"):
            return _CandidateParts((), (), FinishReason.CONTENT_FILTER)
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Gemini API returned a response without candidates.",
            provider="gemini",
            model=model,
        )
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message="Gemini API returned a malformed candidate.",
            provider="gemini",
            model=model,
        )
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        parts = []
    text_parts = tuple(
        str(part["text"])
        for part in parts
        if isinstance(part, Mapping) and isinstance(part.get("text"), str)
    )
    tool_calls = tuple(
        ToolCall(
            id=str(call.get("id") or f"gemini-tool:{index}"),
            name=str(call["name"]),
            arguments=call.get("args") if isinstance(call.get("args"), Mapping) else {},
        )
        for index, part in enumerate(parts)
        if isinstance(part, Mapping)
        and isinstance((call := part.get("functionCall")), Mapping)
        and isinstance(call.get("name"), str)
    )
    return _CandidateParts(
        text_parts=text_parts,
        tool_calls=tool_calls,
        finish_reason=_finish_reason(candidate.get("finishReason")),
    )


def _usage(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(value.get("promptTokenCount", 0) or 0),
        output_tokens=int(value.get("candidatesTokenCount", 0) or 0),
    )


def _finish_reason(value: object) -> FinishReason:
    normalized = str(value or "STOP").upper()
    if normalized in {"MAX_TOKENS", "MODEL_CONTEXT_WINDOW_EXCEEDED"}:
        return FinishReason.LENGTH
    if normalized in {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
    }:
        return FinishReason.CONTENT_FILTER
    if normalized == "MALFORMED_FUNCTION_CALL":
        return FinishReason.ERROR
    return FinishReason.STOP


def _tool_response(content: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {"result": content}
    return value if isinstance(value, Mapping) else {"result": value}


def _model_name(model: str) -> str:
    return _wire.require_text("model", model).removeprefix("models/")


def _model_path(model: str, method: str) -> str:
    return f"models/{quote(_model_name(model), safe='-._')}:{method}"
