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
from financial_research_agent.web import ChatSessionStore, create_app


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
    assert payload["history"]["session_count"] == 0
    assert "secret-value" not in json.dumps(payload)


def test_session_creation_and_retrieval() -> None:
    client = _client()

    created = client.post("/api/sessions").json()["session"]
    retrieved = client.get(f"/api/sessions/{created['id']}").json()["session"]

    assert created["id"].startswith("session_")
    assert retrieved == created
    assert retrieved["messages"] == []
    assert retrieved["summary"] is None


def test_session_list_delete_and_clear() -> None:
    client = _client()
    first = client.post("/api/sessions").json()["session"]
    second = client.post("/api/sessions").json()["session"]

    listed = client.get("/api/sessions").json()["sessions"]
    deleted = client.delete(f"/api/sessions/{first['id']}")
    clear = client.delete("/api/sessions")
    remaining = client.get("/api/sessions").json()["sessions"]

    assert [session["id"] for session in listed] == [second["id"], first["id"]]
    assert deleted.status_code == 200
    assert clear.json()["deleted"] == 1
    assert remaining == []


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


def test_chat_request_uses_bounded_recent_context_and_summary(tmp_path) -> None:
    provider = CapturingProvider()
    registry = ProviderRegistry().register_chat_provider("capture", provider)
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_LLM_PROVIDER": "capture",
            "FRA_LLM_MODEL": "capture-model",
            "FRA_CHAT_HISTORY_RECENT_TURNS": "1",
            "FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS": "500",
        }
    )
    client = _client(settings=settings, registry=registry, use_default_store=True)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    for content in ("first", "second", "third"):
        response = client.post(f"/api/sessions/{session_id}/messages", json={"content": content})
        assert response.status_code == 200

    latest_request = provider.requests[-1]

    assert latest_request.messages[0].role == MessageRole.SYSTEM
    assert latest_request.messages[1].role == MessageRole.SYSTEM
    assert "Earlier conversation summary" in latest_request.messages[1].content
    assert "first" in latest_request.messages[1].content
    assert [message.content for message in latest_request.messages[-3:]] == [
        "second",
        "captured response",
        "third",
    ]


def test_default_app_store_persists_sessions_between_app_instances(tmp_path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    first_client = _client(settings=settings, use_default_store=True)
    session_id = first_client.post("/api/sessions").json()["session"]["id"]

    second_client = _client(settings=settings, use_default_store=True)
    sessions = second_client.get("/api/sessions").json()["sessions"]

    assert [session["id"] for session in sessions] == [session_id]


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
    use_default_store: bool = False,
) -> TestClient:
    return TestClient(
        create_app(
            settings=settings or Settings.from_env({}),
            registry=registry or create_offline_provider_registry(),
            session_store=None if use_default_store else ChatSessionStore(),
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
