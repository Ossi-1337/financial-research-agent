from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.domain import FinancialStatementType
from financial_research_agent.entities import (
    CompanySearchCandidate,
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
    SourceMetadata,
)
from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingIngestionResult,
    FilingSource,
    FilingStore,
)
from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    MessageRole,
    ModelMetadata,
    OfflineTestProvider,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from financial_research_agent.llm.registry import ProviderRegistry, create_offline_provider_registry
from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataError,
    MarketDataErrorCode,
    MarketDataSource,
    MarketDataStore,
    MarketSecurity,
    calculate_price_metrics,
)
from financial_research_agent.orchestration import OrchestratorRunStore
from financial_research_agent.reports import CitedResearchRunStore
from financial_research_agent.retrieval import (
    IndexedChunk,
    LocalVectorIndex,
    RetrievalChunk,
    RetrievalSourceKind,
)
from financial_research_agent.runtime_settings import RuntimeSettingsStore
from financial_research_agent.settings import Settings
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementResult,
    FinancialStatementSource,
    FinancialStatementStore,
    NormalizedFinancialStatement,
)
from financial_research_agent.web import ChatSessionStore, create_app


def test_root_html_and_static_asset_are_served() -> None:
    client = _client()

    root_response = client.get("/")
    css_response = client.get("/static/styles.css")

    assert root_response.status_code == 200
    assert "Financial Research Agent" in root_response.text
    assert 'id="mention-menu"' in root_response.text
    assert 'id="send-button"' in root_response.text
    assert 'id="context-panel"' in root_response.text
    assert 'id="context-source-list"' in root_response.text
    assert 'id="settings-panel"' in root_response.text
    assert 'id="settings-button"' in root_response.text
    assert 'class="composer-action"' in root_response.text
    assert 'id="company-search-form"' not in root_response.text
    assert 'id="selected-company"' not in root_response.text
    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert ".mention-menu[hidden]" in css_response.text
    assert "display: none" in css_response.text
    assert "--accent: #2563eb" in css_response.text
    assert "resize: none" in css_response.text
    assert "overflow-wrap: anywhere" in css_response.text
    assert "html {" in css_response.text
    assert "overflow: hidden" in css_response.text
    assert "height: 100vh" in css_response.text
    assert ".message.assistant" in css_response.text
    assert "width: 100%" in css_response.text
    assert "border-top: 1px solid var(--border)" in css_response.text
    assert ".context-panel[hidden]" in css_response.text
    assert ".context-source-link" in css_response.text
    assert ".citation-list" in css_response.text
    assert ".evidence-snippet" in css_response.text
    assert ".synthesis-report" in css_response.text
    assert ".trace-timeline" in css_response.text
    assert ".settings-panel" in css_response.text


def test_static_script_contains_mention_autocomplete_wiring() -> None:
    client = _client()

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "/api/company-search?query=" in response.text
    assert "mention-chip" in response.text
    assert "Stop response" in response.text
    assert "/messages/stream" in response.text
    assert "appendOptimisticExchange" in response.text
    assert "readNdjsonStream" in response.text
    assert "getReader" in response.text
    assert "renderLoadingIndicator" in response.text
    assert "renderMessageCitations" in response.text
    assert "renderContextPanel" in response.text
    assert "renderSynthesisReport" in response.text
    assert "renderTraceTimeline" in response.text
    assert "loadRunTrace" in response.text
    assert "/synthesis-report" in response.text
    assert "/trace" in response.text
    assert "researchCommand" in response.text
    assert "/api/settings" in response.text
    assert "/api/settings/provider-health" in response.text
    assert "contextSourcesFromMessages" in response.text
    assert "safeExternalUrl" in response.text
    assert "citation-list" in response.text
    assert 'item.className = "loading-row"' in response.text
    assert "message.provider" not in response.text


def test_runtime_settings_endpoint_returns_redacted_provider_management_payload() -> None:
    settings = Settings.from_env(
        {
            "FRA_OPENAI_API_KEY": "secret-value",
            "FRA_ALPHA_VANTAGE_API_KEY": "alpha-secret",
        }
    )
    client = _client(settings=settings)

    response = client.get("/api/settings")
    payload = response.json()
    dumped = json.dumps(payload)

    assert response.status_code == 200
    assert payload["settings"]["provider"]["llm_provider"] == "offline-test"
    assert payload["secrets"]["strategy"] == "environment_only"
    assert payload["secrets"]["plaintext_storage"] == "disabled"
    assert payload["secrets"]["openai_api_key_configured"] is True
    assert any(provider["provider"] == "offline-test" for provider in payload["providers"])
    assert payload["management"]["cache_clear_endpoint"] == "/api/storage/cache"
    assert "secret-value" not in dumped
    assert "alpha-secret" not in dumped


def test_runtime_settings_update_changes_chat_model_without_restart() -> None:
    client = _client()
    session_id = client.post("/api/sessions").json()["session"]["id"]

    settings_response = client.put(
        "/api/settings",
        json={"llm_provider": "offline-test", "llm_model": "custom-offline-model"},
    )
    status = client.get("/api/status").json()
    chat_response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "Use the selected model."},
    )

    assert settings_response.status_code == 200
    assert settings_response.json()["overrides"]["llm_model"] == "custom-offline-model"
    assert status["chat"]["model"] == "custom-offline-model"
    assert chat_response.status_code == 200
    assert chat_response.json()["model"] == "custom-offline-model"


def test_runtime_settings_reject_secret_fields_and_can_reset() -> None:
    client = _client()

    rejected = client.put("/api/settings", json={"openai_api_key": "secret-value"})
    rejected_generic = client.put("/api/settings", json={"api_key": "another-secret"})
    saved = client.put("/api/settings", json={"llm_model": "custom-offline-model"})
    reset = client.delete("/api/settings")

    assert rejected.status_code == 400
    assert "secret-value" not in json.dumps(rejected.json())
    assert rejected_generic.status_code == 400
    assert "another-secret" not in json.dumps(rejected_generic.json())
    assert saved.json()["overrides"]["llm_model"] == "custom-offline-model"
    assert reset.status_code == 200
    assert reset.json()["overrides"] == {}
    assert reset.json()["settings"]["provider"]["llm_model"] == "offline-test"


def test_runtime_provider_health_reports_offline_capabilities() -> None:
    client = _client()

    response = client.get("/api/settings/provider-health?provider=offline-test")
    health = response.json()["provider_health"]

    assert response.status_code == 200
    assert health["provider"] == "offline-test"
    assert health["reachable"] is True
    assert health["status"] == "ok"
    assert "chat" in health["capabilities"]


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
    assert payload["company_search"]["provider"] == "sec"
    assert payload["financial_statements"]["provider"] == "sec-companyfacts"
    assert payload["filings"]["provider"] == "sec-edgar"
    assert payload["retrieval"]["provider"] == "local-vector"
    assert payload["retrieval"]["index"]["record_count"] == 0
    assert payload["retrieval"]["embedding_provider"] == "disabled"
    assert payload["report_runs"]["stored_run_count"] == 0
    assert payload["context_analysis"]["source"] == "explicit_source_items"
    assert payload["context_analysis"]["recommendations"] == "disabled"
    assert payload["orchestration"]["execution_policy"] == "sequential_local_safe"
    assert payload["orchestration"]["stored_run_count"] == 0
    assert payload["orchestration"]["recommendations"] == "disabled"
    assert payload["synthesis"]["source"] == "orchestrator_specialist_handoffs"
    assert payload["synthesis"]["recommendations"] == "disabled"
    assert payload["observability"]["source"] == "stored_orchestrator_runs"
    assert payload["observability"]["hosted_telemetry"] == "disabled"
    assert payload["observability"]["debug_bundle"] == "redacted_local_json"
    assert payload["performance"]["embedding_cache"]["stores_raw_text"] is False
    assert payload["performance"]["prompt_budgets"]["chat"]["max_input_tokens"] == 16_000
    assert [item["id"] for item in payload["performance"]["local_model_profiles"]] == [
        "small",
        "medium",
        "strong",
    ]
    assert payload["interoperability"]["enabled"] is False
    assert payload["interoperability"]["api_key_configured"] is False
    assert payload["storage"]["provider"] == "local-json"
    assert "secret-value" not in json.dumps(payload)


def test_interop_endpoints_are_disabled_by_default() -> None:
    client = _client()

    card = client.get("/.well-known/agent.json")
    mcp = client.post("/api/interop/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert card.status_code == 404
    assert card.json()["detail"]["error"] == "interoperability_disabled"
    assert mcp.status_code == 404
    assert mcp.json()["detail"]["error"] == "interoperability_disabled"


def test_local_interop_agent_card_and_mcp_status_tool_are_read_only() -> None:
    settings = Settings.from_env({"FRA_INTEROP_ENABLED": "true"})
    client = _client(settings=settings)

    card_response = client.get("/.well-known/agent-card.json")
    initialize_response = client.post(
        "/api/interop/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )
    list_response = client.post(
        "/api/interop/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    call_response = client.post(
        "/api/interop/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "financial_research_agent.status", "arguments": {}},
        },
    )

    card = card_response.json()
    tool_text = call_response.json()["result"]["content"][0]["text"]
    tool_payload = json.loads(tool_text)

    assert card_response.status_code == 200
    assert card["name"] == "financial-research-agent"
    assert card["skills"][0]["id"] == "read_sanitized_status"
    assert card["capabilities"]["streaming"] is False
    assert initialize_response.json()["result"]["capabilities"]["tools"]["listChanged"] is False
    assert list_response.json()["result"]["tools"][0]["name"] == "financial_research_agent.status"
    assert tool_payload["app"] == "financial-research-agent"
    assert tool_payload["capabilities"]["recommendations"] == "disabled"
    assert "secret-key" not in json.dumps(card_response.json()).casefold()
    assert "secret" not in tool_text.casefold()


def test_interop_remote_mode_requires_api_key_and_accepts_bearer_or_header() -> None:
    settings = Settings.from_env(
        {
            "FRA_INTEROP_ENABLED": "true",
            "FRA_INTEROP_LOCAL_ONLY": "false",
            "FRA_INTEROP_API_KEY": "secret-key",
        }
    )
    client = _client(settings=settings)

    denied = client.get("/.well-known/agent.json")
    header_allowed = client.post(
        "/api/interop/mcp",
        headers={"X-FRA-Interop-Key": "secret-key"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    bearer_allowed = client.get(
        "/.well-known/agent.json",
        headers={"Authorization": "Bearer secret-key"},
    )

    assert denied.status_code == 401
    assert denied.json()["detail"]["error"] == "invalid_interop_key"
    assert header_allowed.status_code == 200
    assert bearer_allowed.status_code == 200
    assert "secret-key" not in json.dumps(header_allowed.json())


def test_interop_mcp_returns_json_rpc_errors_for_unknown_tool() -> None:
    settings = Settings.from_env({"FRA_INTEROP_ENABLED": "true"})
    client = _client(settings=settings)

    response = client.post(
        "/api/interop/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "bad-tool",
            "method": "tools/call",
            "params": {"name": "shell.exec", "arguments": {}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "bad-tool"
    assert payload["error"]["code"] == -32602


def test_storage_status_endpoint_reports_local_datasets(tmp_path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    cache_path = tmp_path / "cache" / "sec_company_tickers.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({"version": 1, "retrieved_at": "2026-07-04T00:00:00+00:00", "records": []}),
        encoding="utf-8",
    )
    client = _client(settings=settings)

    response = client.get("/api/storage")
    payload = response.json()["storage"]

    assert response.status_code == 200
    assert payload["provider"] == "local-json"
    assert payload["app_home"] == str(tmp_path)
    cache_entries = [
        entry for entry in payload["datasets"] if entry["spec"]["id"] == "company_lookup_cache"
    ]
    assert cache_entries[0]["exists"] is True
    assert cache_entries[0]["record_count"] == 0


def test_storage_migrate_endpoint_creates_local_layout(tmp_path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    client = _client(settings=settings)

    response = client.post("/api/storage/migrate")
    payload = response.json()["result"]

    assert response.status_code == 200
    assert payload["applied_migrations"][0]["id"] == "0001_local_json_storage_layout"
    assert tmp_path.joinpath("data", "storage_migrations.json").exists()


def test_storage_cache_clear_endpoint_removes_cache_without_data(tmp_path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    cache_path = tmp_path / "cache" / "sec_company_tickers.json"
    data_path = tmp_path / "data" / "chat_sessions.json"
    cache_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    cache_path.write_text("{}", encoding="utf-8")
    data_path.write_text("{}", encoding="utf-8")
    client = _client(settings=settings)

    response = client.delete("/api/storage/cache")
    payload = response.json()["result"]

    assert response.status_code == 200
    assert payload["deleted_count"] == 1
    assert not cache_path.exists()
    assert data_path.exists()


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
    assert payload["performance"]["call_kind"] == "chat"
    assert payload["performance"]["provider"] == "offline-test"
    assert payload["performance"]["estimated_cost_usd"] == "0.000000"
    assert payload["assistant_message"]["role"] == "assistant"
    assistant_content = payload["assistant_message"]["content"]
    assert "offline-test response: Summarize Novo Nordisk." in assistant_content
    assert len(payload["session"]["messages"]) == 2
    assert retrieved["messages"] == payload["session"]["messages"]


def test_chat_request_includes_financial_research_system_prompt() -> None:
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
    assert "does not automatically receive live financial data" in system_prompt.content
    assert "must fetch and inspect source data first" in system_prompt.content
    assert "Do not provide buy, sell, or hold recommendations" in system_prompt.content
    assert request.messages[-1].content == "Hello"
    assert request.max_output_tokens is not None


def test_chat_request_accepts_mentions_and_adds_provider_context() -> None:
    provider = CapturingProvider()
    registry = ProviderRegistry().register_chat_provider("capture", provider)
    settings = Settings.from_env({"FRA_LLM_PROVIDER": "capture", "FRA_LLM_MODEL": "capture-model"})
    client = _client(settings=settings, registry=registry)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "content": "Summarize @AAPL.",
            "mentions": [
                {
                    "id": "sec:company:320193",
                    "label": "AAPL",
                    "company_id": "sec:company:320193",
                    "legal_name": "TEST TOOL OUTPUT APPLE INC.",
                    "ticker": "AAPL",
                    "cik": "320193",
                    "source_provider": "sec",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    user_message = payload["session"]["messages"][0]
    assert user_message["mentions"][0]["label"] == "AAPL"
    request = provider.requests[0]
    assert "Resolved @company mentions" in request.messages[1].content
    assert "ticker=AAPL" in request.messages[1].content
    assert "cik=320193" in request.messages[1].content
    assert "not live financial evidence" in request.messages[1].content
    assert request.messages[-1].content == "Summarize @AAPL."


def test_streaming_chat_message_emits_deltas_and_updates_session() -> None:
    provider = CapturingProvider()
    registry = ProviderRegistry().register_chat_provider("capture", provider)
    settings = Settings.from_env({"FRA_LLM_PROVIDER": "capture", "FRA_LLM_MODEL": "capture-model"})
    client = _client(settings=settings, registry=registry)
    session_id = client.post("/api/sessions").json()["session"]["id"]

    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/messages/stream",
        json={"content": "Stream this answer."},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    retrieved = client.get(f"/api/sessions/{session_id}").json()["session"]

    assert response.status_code == 200
    assert [event["type"] for event in events] == ["delta", "delta", "completed"]
    assert "".join(event["delta"] for event in events if event["type"] == "delta") == (
        "captured response"
    )
    assert events[-1]["assistant_message"]["content"] == "captured response"
    assert events[-1]["performance"]["call_kind"] == "streaming_chat"
    assert events[-1]["session"]["messages"] == retrieved["messages"]
    assert provider.requests[0].messages[-1].content == "Stream this answer."


def test_streaming_provider_error_event_does_not_mutate_session() -> None:
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

    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/messages/stream",
        json={"content": "Hello"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    retrieved = client.get(f"/api/sessions/{session_id}").json()["session"]

    assert response.status_code == 200
    assert events == [
        {
            "type": "error",
            "status": 503,
            "detail": {
                "error": "provider_error",
                "code": "provider_unavailable",
                "message": "Local endpoint is unavailable.",
                "provider": "offline-test",
                "model": None,
                "retryable": True,
            },
        }
    ]
    assert retrieved["messages"] == []


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

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "content": "Hello @AAPL",
            "mentions": [
                {
                    "id": "sec:company:320193",
                    "label": "AAPL",
                    "company_id": "sec:company:320193",
                    "legal_name": "TEST TOOL OUTPUT APPLE INC.",
                }
            ],
        },
    )
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


def test_company_search_endpoint_returns_reviewable_candidates() -> None:
    client = _client(company_search_provider=FakeCompanySearchProvider())

    response = client.get("/api/company-search", params={"query": "Novo Nordisk", "limit": 3})
    payload = response.json()["result"]

    assert response.status_code == 200
    assert payload["status"] == "review_required"
    assert payload["source"]["provider"] == "fake-company-search"
    assert payload["candidates"][0]["company"]["legal_name"] == "TEST TOOL OUTPUT NOVO NORDISK"
    assert payload["candidates"][0]["securities"][0]["ticker"] == "NVO"


def test_company_search_errors_map_to_http_status() -> None:
    client = _client(
        company_search_provider=FailingCompanySearchProvider(
            CompanySearchErrorCode.RATE_LIMITED,
        )
    )

    response = client.get("/api/company-search", params={"query": "Novo Nordisk"})

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "rate_limited"


def test_market_data_fetch_endpoint_persists_history(tmp_path) -> None:
    store = MarketDataStore(storage_path=tmp_path / "market_data_price_bars.json")
    client = _client(market_data_provider=FakeMarketDataProvider(), market_data_store=store)

    response = client.post(
        "/api/market-data/history",
        json={
            "symbol": "NVO",
            "security_id": "fixture:security:nvo",
            "currency": "USD",
            "exchange_mic": "XNYS",
            "refresh": True,
        },
    )
    stored = client.get("/api/market-data/history/NVO")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stored"] is False
    assert payload["history"]["metrics"]["latest_close"] == "105"
    assert stored.status_code == 200
    assert stored.json()["stored"] is True
    assert stored.json()["history"]["bars"][-1]["close"] == "105"


def test_market_data_endpoint_uses_cached_history_without_refresh(tmp_path) -> None:
    store = MarketDataStore(storage_path=tmp_path / "market_data_price_bars.json")
    store.save_history(FakeMarketDataProvider.history())
    client = _client(
        market_data_provider=FailingMarketDataProvider(MarketDataErrorCode.AUTHENTICATION_FAILED),
        market_data_store=store,
    )

    response = client.post("/api/market-data/history", json={"symbol": "NVO"})

    assert response.status_code == 200
    assert response.json()["stored"] is True


def test_market_data_errors_map_to_http_status() -> None:
    client = _client(
        market_data_provider=FailingMarketDataProvider(MarketDataErrorCode.AUTHENTICATION_FAILED),
        market_data_store=MarketDataStore(),
    )

    response = client.post("/api/market-data/history", json={"symbol": "NVO", "refresh": True})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "authentication_failed"


def test_financial_statement_fetch_endpoint_persists_result(tmp_path) -> None:
    store = FinancialStatementStore(storage_path=tmp_path / "financial_statements.json")
    client = _client(
        financial_statement_provider=FakeFinancialStatementProvider(),
        financial_statement_store=store,
    )

    response = client.post(
        "/api/financial-statements",
        json={
            "cik": "0000320193",
            "company_id": "fixture:company:apple",
            "legal_name": "TEST TOOL OUTPUT APPLE INC.",
            "refresh": True,
        },
    )
    stored = client.get("/api/financial-statements/320193")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stored"] is False
    assert payload["statements"]["statements"][0]["line_items"]["revenues"] == "1000"
    assert stored.status_code == 200
    assert stored.json()["stored"] is True
    assert stored.json()["statements"]["company"]["cik"] == "320193"


def test_financial_statement_endpoint_uses_cached_result_without_refresh(tmp_path) -> None:
    store = FinancialStatementStore(storage_path=tmp_path / "financial_statements.json")
    store.save_result(FakeFinancialStatementProvider.result())
    client = _client(
        financial_statement_provider=FailingFinancialStatementProvider(
            FinancialStatementErrorCode.PROVIDER_UNAVAILABLE
        ),
        financial_statement_store=store,
    )

    response = client.post("/api/financial-statements", json={"cik": "320193"})

    assert response.status_code == 200
    assert response.json()["stored"] is True


def test_financial_statement_errors_map_to_http_status() -> None:
    client = _client(
        financial_statement_provider=FailingFinancialStatementProvider(
            FinancialStatementErrorCode.NOT_FOUND
        ),
        financial_statement_store=FinancialStatementStore(),
    )

    response = client.post(
        "/api/financial-statements",
        json={"cik": "320193", "refresh": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_filing_ingestion_endpoint_persists_result(tmp_path) -> None:
    store = FilingStore(storage_path=tmp_path / "filings_index.json")
    client = _client(filing_provider=FakeFilingProvider(), filing_store=store)

    response = client.post(
        "/api/filings/ingest",
        json={
            "cik": "0000320193",
            "company_id": "fixture:company:apple",
            "legal_name": "TEST TOOL OUTPUT APPLE INC.",
            "forms": ["10-K"],
            "limit": 1,
            "refresh": True,
        },
    )
    stored = client.get("/api/filings/320193")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stored"] is False
    assert payload["filings"]["filings"][0]["form_type"] == "10-K"
    assert payload["filings"]["chunks"][0]["section_heading"] == "Item 1. Business"
    assert stored.status_code == 200
    assert stored.json()["stored"] is True


def test_filing_ingestion_endpoint_uses_cached_result_without_refresh(tmp_path) -> None:
    store = FilingStore(storage_path=tmp_path / "filings_index.json")
    store.save_result(FakeFilingProvider.result())
    client = _client(
        filing_provider=FailingFilingProvider(FilingErrorCode.PROVIDER_UNAVAILABLE),
        filing_store=store,
    )

    response = client.post("/api/filings/ingest", json={"cik": "320193"})

    assert response.status_code == 200
    assert response.json()["stored"] is True


def test_filing_errors_map_to_http_status() -> None:
    client = _client(
        filing_provider=FailingFilingProvider(FilingErrorCode.UNSUPPORTED_FORMAT),
        filing_store=FilingStore(),
    )

    response = client.post(
        "/api/filings/ingest",
        json={"cik": "320193", "forms": ["10-K"], "refresh": True},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_format"


def test_retrieval_index_endpoint_reports_metadata() -> None:
    client = _client()

    response = client.get("/api/retrieval/index")
    payload = response.json()["index"]

    assert response.status_code == 200
    assert payload["provider"] == "local-vector"
    assert payload["record_count"] == 0


def test_retrieval_index_stored_filings_and_searches_chunks(tmp_path) -> None:
    filing_store = FilingStore(storage_path=tmp_path / "filings_index.json")
    filing_store.save_result(FakeFilingProvider.result())
    retrieval_index = LocalVectorIndex(storage_path=tmp_path / "vector_index.json")
    registry = create_offline_provider_registry().register_embedding_provider(
        "keyword-fixture", KeywordEmbeddingProvider()
    )
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_EMBEDDING_PROVIDER": "keyword-fixture",
            "FRA_EMBEDDING_MODEL": "keyword-model",
        }
    )
    client = _client(
        settings=settings,
        registry=registry,
        filing_store=filing_store,
        retrieval_index=retrieval_index,
    )

    indexed = client.post("/api/retrieval/index/filings", json={"cik": "320193"})
    searched = client.post("/api/retrieval/search", json={"query": "filing", "top_k": 1})

    assert indexed.status_code == 200
    assert indexed.json()["result"]["indexed_count"] == 1
    assert indexed.json()["index"]["record_count"] == 1
    assert searched.status_code == 200
    match = searched.json()["result"]["matches"][0]
    assert match["chunk"]["metadata"]["cik"] == "320193"
    assert match["chunk"]["source_url"] == "https://example.invalid/aapl-20251231.htm"
    assert match["score"] > 0


def test_retrieval_search_empty_index_maps_to_not_found(tmp_path) -> None:
    registry = create_offline_provider_registry().register_embedding_provider(
        "keyword-fixture", KeywordEmbeddingProvider()
    )
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_EMBEDDING_PROVIDER": "keyword-fixture",
        }
    )
    client = _client(settings=settings, registry=registry, retrieval_index=LocalVectorIndex())

    response = client.post("/api/retrieval/search", json={"query": "filing"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "index_empty"


def test_cited_answer_endpoint_stores_run_and_session_citations(tmp_path) -> None:
    provider = CapturingProvider()
    registry = (
        ProviderRegistry()
        .register_chat_provider("capture", provider)
        .register_embedding_provider("keyword-fixture", KeywordEmbeddingProvider())
    )
    index = LocalVectorIndex(storage_path=tmp_path / "vector_index.json")
    index.upsert(
        (
            IndexedChunk(
                chunk=RetrievalChunk(
                    id="retrieval:chunk-1",
                    text="TEST TOOL OUTPUT filing revenue evidence.",
                    source_kind=RetrievalSourceKind.FILING_CHUNK,
                    source_id="filing-chunk-1",
                    source_url="https://example.invalid/aapl-10k.htm",
                    document_id="filing-1",
                    section_heading="Item 7. Management Discussion",
                    metadata={"cik": "320193", "char_start": "42"},
                ),
                embedding=(1.0, 0.0, 0.0),
                embedding_provider="keyword-fixture",
                embedding_model="keyword-model",
                indexed_at=datetime(2026, 7, 5, tzinfo=UTC),
            ),
        )
    )
    report_store = CitedResearchRunStore(storage_path=tmp_path / "report_runs.json")
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_LLM_PROVIDER": "capture",
            "FRA_LLM_MODEL": "capture-model",
            "FRA_EMBEDDING_PROVIDER": "keyword-fixture",
            "FRA_EMBEDDING_MODEL": "keyword-model",
        }
    )
    client = _client(
        settings=settings,
        registry=registry,
        retrieval_index=index,
        report_run_store=report_store,
    )
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(
        f"/api/sessions/{session_id}/cited-answer",
        json={"content": "What does the filing say about revenue?", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assistant = payload["assistant_message"]
    assert assistant["research_run_id"].startswith("research_run_")
    assert assistant["content"].endswith("Sources: [C1]")
    assert assistant["citations"][0]["marker"] == "[C1]"
    assert assistant["citations"][0]["quote_start"] == 42
    assert assistant["evidence_snippets"][0]["text"].startswith("TEST TOOL OUTPUT filing")
    assert payload["research_run"]["citations"][0]["source_url"].endswith("aapl-10k.htm")
    assert "[C1]" in provider.requests[0].messages[-1].content
    stored = client.get(f"/api/research-runs/{assistant['research_run_id']}")
    assert stored.status_code == 200
    assert stored.json()["research_run"]["citations"][0]["id"] == "C1"


def test_cited_answer_missing_evidence_adds_limitation_without_llm_call(tmp_path) -> None:
    provider = CapturingProvider()
    registry = (
        ProviderRegistry()
        .register_chat_provider("capture", provider)
        .register_embedding_provider("keyword-fixture", KeywordEmbeddingProvider())
    )
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_LLM_PROVIDER": "capture",
            "FRA_LLM_MODEL": "capture-model",
            "FRA_EMBEDDING_PROVIDER": "keyword-fixture",
        }
    )
    client = _client(
        settings=settings,
        registry=registry,
        retrieval_index=LocalVectorIndex(),
        report_run_store=CitedResearchRunStore(storage_path=tmp_path / "report_runs.json"),
    )
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(
        f"/api/sessions/{session_id}/cited-answer",
        json={"content": "Find unsupported facts."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["research_run"]["status"] == "limited"
    assert payload["research_run"]["citations"] == []
    assert "could not find stored evidence" in payload["assistant_message"]["content"]
    assert provider.requests == []


def test_session_synthesis_report_endpoint_runs_orchestrator_and_stores_report() -> None:
    client = _client()
    session_id = client.post("/api/sessions").json()["session"]["id"]

    response = client.post(
        f"/api/sessions/{session_id}/synthesis-report",
        json={"query": "Novo Nordisk financial situation", "refresh": True},
    )
    payload = response.json()
    assistant = payload["assistant_message"]
    retrieved = client.get(f"/api/sessions/{session_id}").json()["session"]
    stored_run = client.get(f"/api/orchestrator/runs/{assistant['research_run_id']}").json()
    trace = client.get(f"/api/orchestrator/runs/{assistant['research_run_id']}/trace").json()
    replay = client.post(f"/api/orchestrator/runs/{assistant['research_run_id']}/replay").json()
    debug_bundle = client.get(
        f"/api/orchestrator/runs/{assistant['research_run_id']}/debug-bundle"
    ).json()

    assert response.status_code == 200
    assert payload["provider"] == "orchestrator"
    assert assistant["role"] == "assistant"
    assert assistant["research_run_id"].startswith("orchestrator_run_")
    assert assistant["synthesis_report"]["sections"]["current_situation"]
    assert assistant["synthesis_report"]["scenarios"]["upside"]["direction"] == "upside"
    assert "does not provide buy, sell, hold" in assistant["content"]
    assert retrieved["messages"][-1]["synthesis_report"] == assistant["synthesis_report"]
    assert stored_run["synthesis_report"]["id"] == assistant["synthesis_report"]["id"]
    assert trace["trace"]["run_id"] == assistant["research_run_id"]
    assert trace["trace"]["events"][0]["kind"] == "provider_call"
    assert any(event["kind"] == "agent_output" for event in trace["trace"]["events"])
    assert replay["replay"]["replayable"] is True
    assert replay["replay"]["steps"][0]["mode"] == "stored_result"
    assert debug_bundle["debug_bundle"]["trace"]["run_id"] == assistant["research_run_id"]
    assert "raw provider credentials" in debug_bundle["debug_bundle"]["excluded_items"]


def test_orchestrator_trace_debug_bundle_redacts_failed_run(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path / "home"),
            "FRA_OPENAI_API_KEY": "sk-test-secret",
        }
    )
    client = _client(
        settings=settings,
        company_search_provider=FailingCompanySearchProvider(
            CompanySearchErrorCode.PROVIDER_UNAVAILABLE,
        ),
        orchestrator_run_store=OrchestratorRunStore(
            storage_path=tmp_path / "orchestrator_runs.json",
        ),
    )

    response = client.post(
        "/api/orchestrator/research",
        json={"query": "sk-test-secret company", "refresh": True},
    )
    run = response.json()["run"]
    trace_response = client.get(f"/api/orchestrator/runs/{run['id']}/trace")
    debug_response = client.get(f"/api/orchestrator/runs/{run['id']}/debug-bundle")

    assert response.status_code == 200
    assert run["status"] == "failed"
    assert trace_response.status_code == 200
    trace = trace_response.json()["trace"]
    event = trace["events"][0]
    assert event["kind"] == "provider_call"
    assert event["status"] == "failed"
    assert event["error_code"] == "provider_unavailable"
    dumped_trace = json.dumps(trace)
    dumped_bundle = json.dumps(debug_response.json())
    assert "sk-test-secret" not in dumped_trace
    assert "sk-test-secret" not in dumped_bundle
    assert str(tmp_path / "home") not in dumped_bundle


def _client(
    *,
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
    use_default_store: bool = False,
    company_search_provider=None,
    market_data_provider=None,
    market_data_store=None,
    financial_statement_provider=None,
    financial_statement_store=None,
    filing_provider=None,
    filing_store=None,
    retrieval_index=None,
    report_run_store=None,
    orchestrator_run_store=None,
    runtime_settings_store=None,
) -> TestClient:
    return TestClient(
        create_app(
            settings=settings or Settings.from_env({}),
            registry=registry or create_offline_provider_registry(),
            session_store=None if use_default_store else ChatSessionStore(),
            company_search_provider=company_search_provider or FakeCompanySearchProvider(),
            market_data_provider=market_data_provider or FakeMarketDataProvider(),
            market_data_store=market_data_store or MarketDataStore(),
            financial_statement_provider=(
                financial_statement_provider or FakeFinancialStatementProvider()
            ),
            financial_statement_store=financial_statement_store or FinancialStatementStore(),
            filing_provider=filing_provider or FakeFilingProvider(),
            filing_store=filing_store or FilingStore(),
            retrieval_index=retrieval_index or LocalVectorIndex(),
            report_run_store=report_run_store or CitedResearchRunStore(),
            orchestrator_run_store=orchestrator_run_store or OrchestratorRunStore(),
            runtime_settings_store=runtime_settings_store or RuntimeSettingsStore(),
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

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        response = await self.chat(request)
        yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta="captured")
        yield StreamEvent(event_type=StreamEventType.MESSAGE_DELTA, delta=" response")
        yield StreamEvent(event_type=StreamEventType.COMPLETED, response=response)


class FakeCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        source = SourceMetadata(
            provider="fake-company-search",
            provider_status="test fixture",
            source_url="https://example.invalid/company-search-fixture",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            attribution="test fixture",
        )
        company = ResolvedCompany(
            id="fixture:company:novo",
            legal_name="TEST TOOL OUTPUT NOVO NORDISK",
            identifiers=(
                EntityIdentifier(
                    EntityIdentifierType.TICKER,
                    "NVO",
                    source="fixture",
                ),
            ),
        )
        security = ResolvedSecurity(
            id="fixture:security:nvo",
            company_id=company.id,
            ticker="NVO",
            name=company.legal_name,
        )
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.REVIEW_REQUIRED,
            candidates=(
                CompanySearchCandidate(
                    company=company,
                    securities=(security,),
                    score=90,
                    match_reason=f"limit_{limit}",
                    source=source,
                ),
            ),
            source=source,
        )


class FailingCompanySearchProvider:
    def __init__(self, code: CompanySearchErrorCode) -> None:
        self.code = code

    async def search(self, _query: str, *, limit: int = 10) -> CompanySearchResult:
        raise CompanySearchError(
            code=self.code,
            message=f"company search failed with limit {limit}",
            provider="fake-company-search",
            retryable=True,
        )


class FakeMarketDataProvider:
    async def fetch_daily_prices(
        self,
        security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        return self.history(security=security, warning=f"outputsize={outputsize}")

    async def fetch_quote(self, security: MarketSecurity):
        raise NotImplementedError

    @staticmethod
    def history(
        *,
        security: MarketSecurity | None = None,
        warning: str = "test fixture",
    ) -> HistoricalPriceResult:
        selected_security = security or MarketSecurity(symbol="NVO")
        bars = (
            HistoricalPriceBar(
                security=selected_security,
                priced_at=datetime(2026, 7, 2, tzinfo=UTC).date(),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=1000,
            ),
            HistoricalPriceBar(
                security=selected_security,
                priced_at=datetime(2026, 7, 3, tzinfo=UTC).date(),
                open=Decimal("102"),
                high=Decimal("106"),
                low=Decimal("101"),
                close=Decimal("105"),
                volume=1200,
            ),
        )
        source = MarketDataSource(
            provider="alpha-vantage",
            provider_status="test fixture",
            source_url="https://example.invalid/market-data-fixture",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            data_as_of=datetime(2026, 7, 3, tzinfo=UTC).date(),
            attribution="test fixture",
        )
        return HistoricalPriceResult(
            security=selected_security,
            bars=bars,
            source=source,
            metrics=calculate_price_metrics(bars),
            warnings=(warning,),
        )


class FailingMarketDataProvider:
    def __init__(self, code: MarketDataErrorCode) -> None:
        self.code = code

    async def fetch_daily_prices(
        self,
        _security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        raise MarketDataError(
            code=self.code,
            message=f"market data failed with outputsize {outputsize}",
            provider="alpha-vantage",
            retryable=True,
        )

    async def fetch_quote(self, security: MarketSecurity):
        raise NotImplementedError


class FakeFinancialStatementProvider:
    async def fetch_statements(
        self,
        company: FinancialStatementCompany,
        *,
        fiscal_years: int = 3,
    ) -> FinancialStatementResult:
        return self.result(company=company, warning=f"fiscal_years={fiscal_years}")

    @staticmethod
    def result(
        *,
        company: FinancialStatementCompany | None = None,
        warning: str = "test fixture",
    ) -> FinancialStatementResult:
        selected_company = company or FinancialStatementCompany(
            cik="320193",
            company_id="fixture:company:apple",
            legal_name="TEST TOOL OUTPUT APPLE INC.",
        )
        source = FinancialStatementSource(
            provider="sec-companyfacts",
            provider_status="test fixture",
            source_url="https://example.invalid/sec-companyfacts-fixture",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            data_as_of=datetime(2026, 6, 30, tzinfo=UTC).date(),
            attribution="test fixture",
        )
        period = FinancialStatementPeriod(
            fiscal_year=2025,
            fiscal_period="FY",
            period_type=FinancialStatementPeriodType.ANNUAL,
            period_start=datetime(2024, 7, 1, tzinfo=UTC).date(),
            period_end=datetime(2025, 6, 30, tzinfo=UTC).date(),
            form="10-K",
            accession_number="fixture-accession",
            filed_at=datetime(2026, 6, 30, tzinfo=UTC).date(),
        )
        return FinancialStatementResult(
            company=selected_company,
            statements=(
                NormalizedFinancialStatement(
                    id="fixture:statement:income:2025",
                    company=selected_company,
                    statement_type=FinancialStatementType.INCOME_STATEMENT,
                    period=period,
                    currency="USD",
                    line_items={"revenues": Decimal("1000"), "net_income_loss": Decimal("250")},
                    source=source,
                ),
            ),
            source=source,
            warnings=(warning,),
        )


class FailingFinancialStatementProvider:
    def __init__(self, code: FinancialStatementErrorCode) -> None:
        self.code = code

    async def fetch_statements(
        self,
        _company: FinancialStatementCompany,
        *,
        fiscal_years: int = 3,
    ) -> FinancialStatementResult:
        raise FinancialStatementError(
            code=self.code,
            message=f"financial statement fetch failed with fiscal_years {fiscal_years}",
            provider="sec-companyfacts",
            retryable=True,
        )


class FakeFilingProvider:
    async def ingest_latest(
        self,
        company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult:
        return self.result(company=company, warning=f"forms={','.join(forms)};limit={limit}")

    @staticmethod
    def result(
        *,
        company: FilingCompany | None = None,
        warning: str = "test fixture",
    ) -> FilingIngestionResult:
        selected_company = company or FilingCompany(
            cik="320193",
            company_id="fixture:company:apple",
            legal_name="TEST TOOL OUTPUT APPLE INC.",
        )
        source = FilingSource(
            provider="sec-edgar",
            provider_status="test fixture",
            source_url="https://example.invalid/submissions/CIK0000320193.json",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            data_as_of=datetime(2026, 1, 31, tzinfo=UTC).date(),
            attribution="test fixture",
        )
        filing = FilingDocument(
            id="fixture:filing:10-k",
            company=selected_company,
            form_type="10-K",
            accession_number="0000320193-25-000001",
            filing_date=datetime(2026, 1, 31, tzinfo=UTC).date(),
            report_date=datetime(2025, 12, 31, tzinfo=UTC).date(),
            publication_date=datetime(2026, 1, 31, tzinfo=UTC).date(),
            document_url="https://example.invalid/aapl-20251231.htm",
            source_url="https://example.invalid/submissions/CIK0000320193.json",
            document_format=FilingDocumentFormat.HTML,
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            local_raw_path="fixture/raw/aapl-20251231.htm",
            local_text_path="fixture/text/aapl-20251231.txt",
            source=source,
            chunk_ids=("fixture:filing:10-k:chunk:0",),
        )
        chunk = FilingChunk(
            id="fixture:filing:10-k:chunk:0",
            filing_id=filing.id,
            chunk_index=0,
            text="TEST TOOL OUTPUT filing chunk",
            char_start=0,
            char_end=29,
            section_heading="Item 1. Business",
            source_url=filing.document_url,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            metadata={"fixture": "true"},
        )
        return FilingIngestionResult(
            company=selected_company,
            filings=(filing,),
            chunks=(chunk,),
            source=source,
            warnings=(warning,),
        )


class FailingFilingProvider:
    def __init__(self, code: FilingErrorCode) -> None:
        self.code = code

    async def ingest_latest(
        self,
        _company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult:
        raise FilingError(
            code=self.code,
            message=f"filing ingestion failed with forms {','.join(forms)} and limit {limit}",
            provider="sec-edgar",
            retryable=True,
        )


class KeywordEmbeddingProvider:
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider="keyword-fixture",
            model="keyword-model",
            capabilities=(ProviderCapability.EMBEDDINGS,),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=tuple(_keyword_vector(text) for text in request.input_texts),
            provider="keyword-fixture",
            model=request.model or "keyword-model",
            usage=TokenUsage(input_tokens=sum(len(text.split()) for text in request.input_texts)),
        )


def _keyword_vector(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    return (
        float(lowered.count("filing")),
        float(lowered.count("risk")),
        float(lowered.count("cash")),
    )
