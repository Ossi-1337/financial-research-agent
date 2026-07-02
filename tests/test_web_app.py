from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ModelMetadata,
    OfflineTestProvider,
    ProviderError,
    ProviderErrorCode,
)
from financial_research_agent.llm.registry import ProviderRegistry, create_offline_provider_registry
from financial_research_agent.settings import Settings
from financial_research_agent.web import create_app


def test_root_html_and_static_asset_are_served() -> None:
    client = _client()

    root_response = client.get("/")
    css_response = client.get("/static/styles.css")

    assert root_response.status_code == 200
    assert "Financial Research Agent" in root_response.text
    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]


def test_status_returns_chat_provider_without_secrets() -> None:
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "offline-test",
            "FRA_LLM_MODEL": "offline-test",
            "FRA_OPENAI_API_KEY": "secret-value",
        }
    )
    client = _client(settings=settings)

    response = client.get("/api/status")
    payload = response.json()

    assert response.status_code == 200
    assert payload["chat"]["provider"] == "offline-test"
    assert payload["chat"]["model"] == "offline-test"
    assert "secret-value" not in json.dumps(payload)


def test_session_creation_and_retrieval() -> None:
    client = _client()

    created = client.post("/api/sessions").json()["session"]
    retrieved = client.get(f"/api/sessions/{created['id']}").json()["session"]

    assert created["id"].startswith("session_")
    assert retrieved == created
    assert retrieved["messages"] == []


def test_unknown_session_returns_404() -> None:
    client = _client()

    response = client.get("/api/sessions/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "session_not_found"


def test_empty_message_is_rejected() -> None:
    client = _client()
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "   "})

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "message_content_required"


def test_chat_message_uses_offline_provider_and_updates_session() -> None:
    client = _client()
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Summarize Novo Nordisk."},
    )
    payload = response.json()
    retrieved = client.get(f"/api/sessions/{session_id}").json()["session"]

    assert response.status_code == 200
    assert payload["provider"] == "offline-test"
    assert payload["model"] == "offline-test"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] > 0
    assert payload["assistant_message"]["role"] == "assistant"
    assistant_content = payload["assistant_message"]["content"]
    assert "offline-test response: Summarize Novo Nordisk." in assistant_content
    assert len(payload["session"]["messages"]) == 2
    assert retrieved["messages"] == payload["session"]["messages"]


def test_chat_request_includes_milestone_10_system_prompt() -> None:
    provider = CapturingProvider()
    registry = ProviderRegistry().register_chat_provider("capture", provider)
    settings = Settings.from_env({"FRA_LLM_PROVIDER": "capture", "FRA_LLM_MODEL": "capture-model"})
    client = _client(settings=settings, registry=registry)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Hello"})

    assert response.status_code == 200
    request = provider.requests[0]
    system_prompt = request.messages[0]
    assert system_prompt.role == MessageRole.SYSTEM
    assert "no live financial data" in system_prompt.content
    assert "Do not provide buy, sell, or hold recommendations" in system_prompt.content
    assert request.messages[-1].content == "Hello"


def test_provider_error_maps_to_http_error_and_does_not_mutate_session() -> None:
    error = ProviderError(
        code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
        message="Local endpoint is unavailable.",
        provider="offline-test",
        retryable=True,
    )
    registry = ProviderRegistry().register_chat_provider(
        "offline-test",
        OfflineTestProvider(fail_with=error),
    )
    client = _client(registry=registry)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Hello"})
    retrieved = client.get(f"/api/sessions/{session_id}").json()["session"]

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "provider_unavailable"
    assert retrieved["messages"] == []


@pytest.mark.parametrize(
    ("code", "expected_status"),
    (
        (ProviderErrorCode.AUTHENTICATION_FAILED, 401),
        (ProviderErrorCode.RATE_LIMITED, 429),
        (ProviderErrorCode.TIMEOUT, 504),
        (ProviderErrorCode.INVALID_REQUEST, 400),
        (ProviderErrorCode.UNSUPPORTED_FEATURE, 400),
        (ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED, 400),
        (ProviderErrorCode.MALFORMED_RESPONSE, 503),
    ),
)
def test_provider_error_status_mapping(code: ProviderErrorCode, expected_status: int) -> None:
    error = ProviderError(code=code, message="Provider failed.", provider="offline-test")
    registry = ProviderRegistry().register_chat_provider(
        "offline-test",
        OfflineTestProvider(fail_with=error),
    )
    client = _client(registry=registry)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Hello"})

    assert response.status_code == expected_status


def _client(
    *,
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            settings=settings or Settings.from_env({}),
            registry=registry or create_offline_provider_registry(),
        )
    )


class CapturingProvider:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(provider="capture", model="capture-model")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="captured response"),
            provider="capture",
            model=request.model or "capture-model",
        )
