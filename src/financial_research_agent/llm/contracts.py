from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class StreamEventType(StrEnum):
    MESSAGE_DELTA = "message_delta"
    TOOL_CALL = "tool_call"
    STRUCTURED_OUTPUT = "structured_output"
    COMPLETED = "completed"
    ERROR = "error"


class ProviderCapability(StrEnum):
    CHAT = "chat"
    TOOL_CALLS = "tool_calls"
    STRUCTURED_OUTPUT = "structured_output"
    EMBEDDINGS = "embeddings"
    STREAMING = "streaming"
    TOKEN_ACCOUNTING = "token_accounting"


class ProviderErrorCode(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    MALFORMED_RESPONSE = "malformed_response"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", MessageRole(self.role))
        object.__setattr__(self, "content", _string_value("content", self.content))
        object.__setattr__(self, "name", _optional_text(self.name))
        object.__setattr__(self, "tool_call_id", _optional_text(self.tool_call_id))
        object.__setattr__(self, "tool_calls", _tool_call_tuple("tool_calls", self.tool_calls))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "input_schema", _freeze_mapping("input_schema", self.input_schema))


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "arguments", _freeze_mapping("arguments", self.arguments))


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    format_type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: Mapping[str, Any] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "format_type", ResponseFormatType(self.format_type))
        object.__setattr__(self, "name", _optional_text(self.name))
        if self.json_schema is not None:
            object.__setattr__(
                self,
                "json_schema",
                _freeze_mapping("json_schema", self.json_schema),
            )
        if self.format_type == ResponseFormatType.JSON_SCHEMA and self.json_schema is None:
            raise ValueError("json_schema is required for json_schema response format")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("input_tokens", self.input_tokens)
        _require_non_negative_int("output_tokens", self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    provider: str
    model: str
    capabilities: tuple[ProviderCapability, ...] = ()
    context_window: int | None = None
    max_output_tokens: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _require_text("model", self.model))
        object.__setattr__(self, "capabilities", _capability_tuple(self.capabilities))
        _require_optional_positive_int("context_window", self.context_window)
        _require_optional_positive_int("max_output_tokens", self.max_output_tokens)
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    def supports(self, capability: ProviderCapability | str) -> bool:
        return ProviderCapability(capability) in self.capabilities


@dataclass(frozen=True, slots=True)
class ProviderHealth:
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "base_url", _require_text("base_url", self.base_url))
        object.__setattr__(self, "model", _require_text("model", self.model))
        object.__setattr__(self, "status", _require_text("status", self.status))
        object.__setattr__(
            self,
            "available_models",
            _text_tuple("available_models", self.available_models),
        )
        object.__setattr__(self, "capabilities", _capability_tuple(self.capabilities))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(self, "error", _optional_text(self.error))

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
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    model: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    response_format: ResponseFormat = field(default_factory=ResponseFormat)
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _message_tuple("messages", self.messages))
        object.__setattr__(self, "model", _optional_text(self.model))
        object.__setattr__(self, "tools", _tool_definition_tuple("tools", self.tools))
        if not isinstance(self.response_format, ResponseFormat):
            raise ValueError("response_format must be a ResponseFormat")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        _require_optional_positive_int("max_output_tokens", self.max_output_tokens)
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class ChatResponse:
    message: ChatMessage
    provider: str
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: tuple[ToolCall, ...] = ()
    structured_output: Mapping[str, Any] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.message, ChatMessage):
            raise ValueError("message must be a ChatMessage")
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _require_text("model", self.model))
        object.__setattr__(self, "finish_reason", FinishReason(self.finish_reason))
        object.__setattr__(self, "tool_calls", _tool_call_tuple("tool_calls", self.tool_calls))
        if self.structured_output is not None:
            object.__setattr__(
                self,
                "structured_output",
                _freeze_mapping("structured_output", self.structured_output),
            )
        if not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be a TokenUsage")
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    input_texts: tuple[str, ...]
    model: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_texts",
            _required_text_tuple("input_texts", self.input_texts),
        )
        object.__setattr__(self, "model", _optional_text(self.model))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    embeddings: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embeddings", _embedding_tuple("embeddings", self.embeddings))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _require_text("model", self.model))
        if not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be a TokenUsage")
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(slots=True)
class ProviderError(Exception):
    code: ProviderErrorCode
    message: str
    provider: str
    model: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ProviderErrorCode(self.code))
        object.__setattr__(self, "message", _require_text("message", self.message))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))
        Exception.__init__(self, self.message)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event_type: StreamEventType
    delta: str | None = None
    message: ChatMessage | None = None
    tool_call: ToolCall | None = None
    structured_output: Mapping[str, Any] | None = None
    response: ChatResponse | None = None
    error: ProviderError | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", StreamEventType(self.event_type))
        object.__setattr__(self, "delta", _optional_string_value("delta", self.delta))
        if self.message is not None and not isinstance(self.message, ChatMessage):
            raise ValueError("message must be a ChatMessage")
        if self.tool_call is not None and not isinstance(self.tool_call, ToolCall):
            raise ValueError("tool_call must be a ToolCall")
        if self.structured_output is not None:
            object.__setattr__(
                self,
                "structured_output",
                _freeze_mapping("structured_output", self.structured_output),
            )
        if self.response is not None and not isinstance(self.response, ChatResponse):
            raise ValueError("response must be a ChatResponse")
        if self.error is not None and not isinstance(self.error, ProviderError):
            raise ValueError("error must be a ProviderError")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    retryable_error_codes: tuple[ProviderErrorCode, ...] = (
        ProviderErrorCode.PROVIDER_UNAVAILABLE,
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.TIMEOUT,
    )

    def __post_init__(self) -> None:
        _require_positive_int("max_attempts", self.max_attempts)
        object.__setattr__(
            self,
            "retryable_error_codes",
            _provider_error_code_tuple("retryable_error_codes", self.retryable_error_codes),
        )

    def is_retryable(self, error: ProviderError) -> bool:
        return error.retryable or error.code in self.retryable_error_codes

    def should_retry(self, error: ProviderError, attempt_number: int) -> bool:
        _require_positive_int("attempt_number", attempt_number)
        return attempt_number < self.max_attempts and self.is_retryable(error)


@runtime_checkable
class ChatProvider(Protocol):
    @property
    def metadata(self) -> ModelMetadata: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> ModelMetadata: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


@runtime_checkable
class HealthCheckProvider(Protocol):
    async def check_health(self) -> ProviderHealth: ...


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _string_value(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _optional_string_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _string_value(name, value)


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _freeze_mapping(name: str, values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {_require_text(f"{name}.key", key): _freeze_value(value) for key, value in values.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping("value", value)
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Sequence):
        return tuple(_freeze_value(item) for item in value)
    return value


def _message_tuple(name: str, values: Iterable[ChatMessage]) -> tuple[ChatMessage, ...]:
    messages = tuple(values)
    if not messages:
        raise ValueError(f"{name} must contain at least one message")
    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessage):
            raise ValueError(f"{name}[{index}] must be a ChatMessage")
    return messages


def _tool_definition_tuple(
    name: str, values: Iterable[ToolDefinition]
) -> tuple[ToolDefinition, ...]:
    tools = tuple(values)
    for index, tool in enumerate(tools):
        if not isinstance(tool, ToolDefinition):
            raise ValueError(f"{name}[{index}] must be a ToolDefinition")
    return tools


def _tool_call_tuple(name: str, values: Iterable[ToolCall]) -> tuple[ToolCall, ...]:
    tool_calls = tuple(values)
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, ToolCall):
            raise ValueError(f"{name}[{index}] must be a ToolCall")
    return tool_calls


def _required_text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    result = tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _capability_tuple(
    values: Iterable[ProviderCapability | str],
) -> tuple[ProviderCapability, ...]:
    return tuple(ProviderCapability(value) for value in values)


def _provider_error_code_tuple(
    name: str, values: Iterable[ProviderErrorCode | str]
) -> tuple[ProviderErrorCode, ...]:
    codes = tuple(ProviderErrorCode(value) for value in values)
    if not codes:
        raise ValueError(f"{name} must contain at least one code")
    return codes


def _embedding_tuple(name: str, values: Iterable[Iterable[float]]) -> tuple[tuple[float, ...], ...]:
    embeddings = tuple(tuple(float(value) for value in embedding) for embedding in values)
    if not embeddings:
        raise ValueError(f"{name} must contain at least one embedding")
    for index, embedding in enumerate(embeddings):
        if not embedding:
            raise ValueError(f"{name}[{index}] must contain at least one value")
    return embeddings


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        _require_positive_int(name, value)
