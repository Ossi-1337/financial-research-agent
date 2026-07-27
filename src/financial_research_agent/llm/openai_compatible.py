from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from financial_research_agent.llm.contracts import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingResponse,
    FinishReason,
    MessageRole,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ResponseFormatType,
    TokenUsage,
    ToolCall,
)


@dataclass(slots=True)
class AsyncClientContext:
    client: httpx.AsyncClient
    close: bool

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, *_exc_info: object) -> None:
        if self.close:
            await self.client.aclose()


def default_capabilities() -> tuple[ProviderCapability, ...]:
    return (
        ProviderCapability.CHAT,
        ProviderCapability.STREAMING,
        ProviderCapability.EMBEDDINGS,
        ProviderCapability.TOKEN_ACCOUNTING,
        ProviderCapability.TOOL_CALLS,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def chat_request_payload(request: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
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
                    "parameters": mutable_json_value(tool.input_schema),
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
                "schema": mutable_json_value(request.response_format.json_schema or {}),
            },
        }
    return payload


def chat_response_from_payload(
    data: Mapping[str, Any],
    provider: str,
    model: str,
    request: ChatRequest,
    service_label: str,
) -> ChatResponse:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned a chat response without choices.",
            provider=provider,
            model=model,
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned a malformed chat choice.",
            provider=provider,
            model=model,
        )
    message_payload = choice.get("message", {})
    if not isinstance(message_payload, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned a malformed chat message.",
            provider=provider,
            model=model,
        )
    content = message_payload.get("content") or ""
    if not isinstance(content, str):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned non-string chat content.",
            provider=provider,
            model=model,
        )
    tool_calls = tool_calls_from_payload(message_payload, provider)
    structured_output = structured_output_from_content(
        content,
        request,
        provider,
        model,
        service_label,
    )
    finish_reason = _finish_reason_from_value(
        choice.get("finish_reason"),
        provider,
        model,
        service_label,
    )

    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        provider=provider,
        model=str(data.get("model") or model),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        structured_output=structured_output,
        usage=usage_from_payload(data.get("usage")),
    )


def structured_output_from_content(
    content: str,
    request: ChatRequest,
    provider: str,
    model: str,
    service_label: str,
) -> Mapping[str, Any] | None:
    if request.response_format.format_type == ResponseFormatType.TEXT:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned malformed structured output JSON.",
            provider=provider,
            model=model,
        ) from exc
    if not isinstance(data, Mapping):
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned non-object structured output JSON.",
            provider=provider,
            model=model,
        )
    return data


def embedding_response_from_payload(
    data: Mapping[str, Any],
    provider: str,
    model: str,
    service_label: str,
) -> EmbeddingResponse:
    values = data.get("data")
    if not isinstance(values, list) or not values:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned an embedding response without data.",
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
                message=f"{service_label} returned a malformed embedding.",
                provider=provider,
                model=model,
            )
        embeddings.append(tuple(float(value) for value in embedding))
    return EmbeddingResponse(
        embeddings=tuple(embeddings),
        provider=provider,
        model=str(data.get("model") or model),
        usage=usage_from_payload(data.get("usage")),
    )


def tool_calls_from_payload(payload: Mapping[str, Any], provider: str) -> tuple[ToolCall, ...]:
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
        arguments = _tool_arguments(function.get("arguments"), provider)
        tool_calls.append(
            ToolCall(
                id=str(value.get("id") or f"tool-call:{index}"),
                name=name,
                arguments=arguments,
            )
        )
    return tuple(tool_calls)


def usage_from_payload(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0),
        output_tokens=int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0),
    )


def model_ids_from_models_payload(data: Mapping[str, Any]) -> tuple[str, ...]:
    values = data.get("data", [])
    if not isinstance(values, list):
        return ()
    return tuple(
        model_id
        for item in values
        if isinstance(item, Mapping)
        for model_id in (_model_id_from_payload(item),)
        if model_id is not None
    )


async def stream_json_lines(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    json_payload: Mapping[str, Any],
    provider: str,
    model: str,
    service_label: str,
) -> AsyncIterator[Mapping[str, Any]]:
    try:
        async with client.stream(method, path, json=json_payload) as response:
            if response.status_code >= 400:
                raise_http_error(response, provider, model, service_label)
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
                        message=f"{service_label} returned malformed SSE JSON from {path}.",
                        provider=provider,
                        model=model,
                    ) from exc
                if not isinstance(data, Mapping):
                    raise ProviderError(
                        code=ProviderErrorCode.MALFORMED_RESPONSE,
                        message=f"{service_label} returned non-object SSE JSON from {path}.",
                        provider=provider,
                        model=model,
                    )
                yield data
    except httpx.TimeoutException as exc:
        raise provider_error_from_timeout(exc, provider, model, service_label) from exc
    except httpx.RequestError as exc:
        raise provider_error_from_request_error(exc, provider, model, service_label) from exc


def raise_http_error(
    response: httpx.Response,
    provider: str,
    model: str | None,
    service_label: str,
) -> None:
    status_code = response.status_code
    error_message, provider_code = _error_details_from_response(response)
    normalized_message = error_message.lower()
    if provider_code in {"context_length_exceeded", "request_too_large"} or any(
        marker in normalized_message
        for marker in (
            "context length",
            "context window",
            "input token count exceeds",
            "prompt is too long",
        )
    ):
        code = ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED
        retryable = False
    elif "not supported" in normalized_message or "unsupported" in normalized_message:
        code = ProviderErrorCode.UNSUPPORTED_FEATURE
        retryable = False
    elif status_code in (400, 422):
        code = ProviderErrorCode.INVALID_REQUEST
        retryable = False
    elif status_code in (401, 403):
        code = ProviderErrorCode.AUTHENTICATION_FAILED
        retryable = False
    elif status_code in (408, 504):
        code = ProviderErrorCode.TIMEOUT
        retryable = True
    elif status_code in (409,):
        code = ProviderErrorCode.PROVIDER_UNAVAILABLE
        retryable = True
    elif status_code == 429:
        code = ProviderErrorCode.RATE_LIMITED
        retryable = True
    elif status_code >= 500:
        code = ProviderErrorCode.PROVIDER_UNAVAILABLE
        retryable = True
    else:
        code = ProviderErrorCode.PROVIDER_UNAVAILABLE
        retryable = False
    detail = f": {error_message}" if error_message else ""
    raise ProviderError(
        code=code,
        message=f"{service_label} returned HTTP {status_code}{detail}.",
        provider=provider,
        model=model,
        retryable=retryable,
    )


def provider_error_from_timeout(
    error: httpx.TimeoutException,
    provider: str,
    model: str,
    service_label: str,
) -> ProviderError:
    return ProviderError(
        code=ProviderErrorCode.TIMEOUT,
        message=f"{service_label} timed out: {error}",
        provider=provider,
        model=model,
        retryable=True,
    )


def provider_error_from_request_error(
    error: httpx.RequestError,
    provider: str,
    model: str,
    service_label: str,
) -> ProviderError:
    return ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message=f"{service_label} is unavailable: {error}",
        provider=provider,
        model=model,
        retryable=True,
    )


def mutable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [mutable_json_value(item) for item in value]
    return value


def normalize_base_url(value: str) -> str:
    return require_text("base_url", value).rstrip("/") + "/"


def require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_payload(tool_call) for tool_call in message.tool_calls]
    return payload


def _tool_call_payload(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(mutable_json_value(tool_call.arguments), sort_keys=True),
        },
    }


def _finish_reason_from_value(
    value: object,
    provider: str,
    model: str,
    service_label: str,
) -> FinishReason:
    finish_reason = value or FinishReason.STOP.value
    try:
        return FinishReason(finish_reason)
    except ValueError as exc:
        raise ProviderError(
            code=ProviderErrorCode.MALFORMED_RESPONSE,
            message=f"{service_label} returned unsupported finish_reason: {finish_reason!r}.",
            provider=provider,
            model=model,
        ) from exc


def _tool_arguments(value: object, provider: str) -> Mapping[str, Any]:
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
                message="Provider returned malformed tool call arguments.",
                provider=provider,
            ) from exc
        if isinstance(data, Mapping):
            return data
    raise ProviderError(
        code=ProviderErrorCode.MALFORMED_RESPONSE,
        message="Provider returned non-object tool call arguments.",
        provider=provider,
    )


def _model_id_from_payload(value: Mapping[str, Any]) -> str | None:
    model_id = value.get("id", value.get("model", value.get("name")))
    return model_id if isinstance(model_id, str) and model_id.strip() else None


def _error_details_from_response(response: httpx.Response) -> tuple[str, str | None]:
    try:
        data = response.json()
    except ValueError:
        return "", None
    if not isinstance(data, Mapping):
        return "", None
    error = data.get("error")
    if not isinstance(error, Mapping):
        return "", None
    message = error.get("message")
    code = error.get("code", error.get("type"))
    return (
        message if isinstance(message, str) else "",
        code if isinstance(code, str) else None,
    )
