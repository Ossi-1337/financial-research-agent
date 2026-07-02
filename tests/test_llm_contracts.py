from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    ResponseFormat,
    ResponseFormatType,
    RetryPolicy,
    TokenUsage,
    ToolDefinition,
)


def test_chat_contracts_are_immutable_and_provider_neutral() -> None:
    message = ChatMessage(
        role="user",
        content="Analyze Novo Nordisk",
        metadata={"source": "test"},
    )
    tool = ToolDefinition(
        name="lookup_company",
        description="Look up a company.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )
    request = ChatRequest(messages=[message], tools=[tool])

    assert message.role == MessageRole.USER
    assert dict(message.metadata) == {"source": "test"}
    assert request.messages == (message,)
    assert request.tools == (tool,)
    assert dict(tool.input_schema)["type"] == "object"

    with pytest.raises(FrozenInstanceError):
        message.content = "Changed"  # type: ignore[misc]

    with pytest.raises(TypeError):
        tool.input_schema["type"] = "changed"  # type: ignore[index]

    with pytest.raises(TypeError):
        tool.input_schema["properties"]["query"] = {"type": "number"}  # type: ignore[index]


def test_contract_validation_rejects_invalid_required_values() -> None:
    with pytest.raises(ValueError, match="messages"):
        ChatRequest(messages=[])

    with pytest.raises(ValueError, match="name is required"):
        ToolDefinition(name=" ", description="Lookup.")

    with pytest.raises(ValueError, match="json_schema is required"):
        ResponseFormat(format_type=ResponseFormatType.JSON_SCHEMA)

    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage(input_tokens=-1)


def test_model_metadata_and_retry_policy_classify_capabilities_and_errors() -> None:
    metadata = ModelMetadata(
        provider="offline-test",
        model="offline-test",
        capabilities=[ProviderCapability.CHAT, "streaming"],
        context_window=8192,
    )
    retryable_error = ProviderError(
        code=ProviderErrorCode.RATE_LIMITED,
        message="Rate limited.",
        provider="online-test",
        model="model-a",
    )
    invalid_error = ProviderError(
        code=ProviderErrorCode.INVALID_REQUEST,
        message="Bad request.",
        provider="online-test",
        model="model-a",
    )
    retry_policy = RetryPolicy(max_attempts=3)

    assert metadata.supports(ProviderCapability.CHAT)
    assert metadata.supports("streaming")
    assert retry_policy.is_retryable(retryable_error)
    assert retry_policy.should_retry(retryable_error, attempt_number=1)
    assert not retry_policy.should_retry(retryable_error, attempt_number=3)
    assert not retry_policy.is_retryable(invalid_error)


def test_provider_error_codes_cover_milestone_failure_model() -> None:
    assert {
        ProviderErrorCode.PROVIDER_UNAVAILABLE,
        ProviderErrorCode.AUTHENTICATION_FAILED,
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.TIMEOUT,
        ProviderErrorCode.INVALID_REQUEST,
        ProviderErrorCode.UNSUPPORTED_FEATURE,
        ProviderErrorCode.MALFORMED_RESPONSE,
        ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED,
    } == set(ProviderErrorCode)
