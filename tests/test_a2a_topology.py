from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from a2a.types import AgentCard
from fastapi.testclient import TestClient
from google.protobuf.json_format import ParseDict

from financial_research_agent.a2a import (
    A2AResearchRuntime,
    A2AResearchStepDispatcher,
    SQLiteA2ATaskStore,
    create_a2a_app,
    create_default_a2a_runtime,
)
from financial_research_agent.credentials import KeyringCredentialStore
from financial_research_agent.orchestration import (
    AgentEndpoint,
    AgentHandoff,
    AgentRole,
    DelegationRequest,
    HandoffConfidence,
    OrchestratorHandoffStatus,
    OrchestratorStepKind,
)
from financial_research_agent.persistence import create_persistence
from financial_research_agent.runtime_settings import RuntimeSettingsOverrides
from financial_research_agent.settings import Settings

NOW = datetime(2026, 7, 27, 12, tzinfo=UTC)
A2A_HEADERS = {"A2A-Version": "1.0"}


class StubSpecialistService:
    async def execute(self, request: DelegationRequest) -> AgentHandoff:
        return AgentHandoff(
            id=f"handoff:{request.step_id}",
            step_id=request.step_id,
            kind=request.expected_kind,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
            output={"analysis": {"fixture": "TEST TOOL OUTPUT"}},
            evidence_ids=("evidence:test:1",),
            confidence=HandoffConfidence.HIGH,
        )


class RetryOnceTransport(httpx.AsyncBaseTransport):
    def __init__(self, app: object) -> None:
        self._inner = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        self.message_ids: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/message:send":
            payload = json.loads(request.content)
            self.message_ids.append(str(payload["message"]["messageId"]))
            if len(self.message_ids) == 1:
                return httpx.Response(503, request=request)
        return await self._inner.handle_async_request(request)


def test_topology_contracts_are_immutable_and_validate_urls() -> None:
    endpoint = AgentEndpoint(
        role=AgentRole.STOCK,
        service_id="stock",
        base_url="http://127.0.0.1:8003/",
        skill_id="stock_price_analysis",
    )
    request = _stock_request()

    assert endpoint.base_url == "http://127.0.0.1:8003"
    assert request.to_dict()["schema_version"] == 1
    with pytest.raises(FrozenInstanceError):
        endpoint.service_id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="http"):
        AgentEndpoint(
            role=AgentRole.STOCK,
            service_id="stock",
            base_url="file:///tmp/stock",
            skill_id="stock_price_analysis",
        )


@pytest.mark.parametrize(
    ("role", "skill_id"),
    [
        (AgentRole.FINANCIAL_REPORT, "financial_report_analysis"),
        (AgentRole.STOCK, "stock_price_analysis"),
        (AgentRole.CONTEXT, "context_analysis"),
        (AgentRole.SYNTHESIS, "research_synthesis"),
    ],
)
def test_specialist_agent_cards_declare_one_structured_skill(
    tmp_path: Path,
    role: AgentRole,
    skill_id: str,
) -> None:
    settings, runtime = _runtime(tmp_path, role)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime, role=role)) as client:
        response = client.get("/.well-known/agent-card.json")
        card = ParseDict(response.json(), AgentCard())

    assert response.status_code == 200
    assert [skill.id for skill in card.skills] == [skill_id]
    assert list(card.default_input_modes) == ["application/json"]
    assert list(card.skills[0].input_modes) == ["application/json"]


def test_specialist_rejects_text_and_returns_handoff_artifact(tmp_path: Path) -> None:
    role = AgentRole.STOCK
    settings, runtime = _runtime(tmp_path, role)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime, role=role)) as client:
        rejected = client.post(
            "/message:send",
            headers=A2A_HEADERS,
            json={
                "message": {
                    "messageId": "text-not-allowed",
                    "role": "ROLE_USER",
                    "parts": [{"text": "run stock analysis"}],
                }
            },
        )
        completed = client.post(
            "/message:send",
            headers=A2A_HEADERS,
            json=_delegation_message("stock-delegation", _stock_request()),
        )
        forbidden = client.post(
            "/message:send",
            headers=A2A_HEADERS,
            json=_delegation_message(
                "stock-forbidden",
                DelegationRequest(
                    role=AgentRole.STOCK,
                    run_id="orchestrator_run_test",
                    step_id="stock_price_analysis",
                    correlation_id="orchestrator_run_test",
                    expected_kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
                    payload={"ticker": "TEST", "provider": "client-controlled"},
                ),
            ),
        )

    assert rejected.json()["task"]["status"]["state"] == "TASK_STATE_REJECTED"
    assert forbidden.json()["task"]["status"]["state"] == "TASK_STATE_REJECTED"
    task = completed.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["metadata"]["kind"] == "specialist_handoff"


def test_official_client_dispatcher_validates_and_parses_specialist_artifact(
    tmp_path: Path,
) -> None:
    role = AgentRole.STOCK
    settings, runtime = _runtime(tmp_path, role)
    app = create_a2a_app(settings=settings, runtime=runtime, role=role)
    endpoint = AgentEndpoint(
        role=role,
        service_id="stock",
        base_url="http://specialist.test",
        skill_id="stock_price_analysis",
    )
    dispatcher = A2AResearchStepDispatcher(
        endpoints={role: endpoint},
        max_attempts=1,
        transport=httpx.ASGITransport(app=app),
    )

    result = asyncio.run(dispatcher.dispatch(_stock_request()))

    assert result.handoff.status == OrchestratorHandoffStatus.SUCCEEDED
    assert result.handoff.execution is not None
    assert result.handoff.execution.agent_role == "stock"
    assert result.handoff.execution.remote_task_id == result.remote_task_id


def test_retry_reuses_deterministic_message_id(tmp_path: Path) -> None:
    role = AgentRole.STOCK
    settings, runtime = _runtime(tmp_path, role)
    app = create_a2a_app(settings=settings, runtime=runtime, role=role)
    transport = RetryOnceTransport(app)
    dispatcher = A2AResearchStepDispatcher(
        endpoints={
            role: AgentEndpoint(
                role=role,
                service_id="stock",
                base_url="http://specialist.test",
                skill_id="stock_price_analysis",
            )
        },
        max_attempts=2,
        transport=transport,
    )

    result = asyncio.run(dispatcher.dispatch(_stock_request()))

    assert result.handoff.status == OrchestratorHandoffStatus.SUCCEEDED
    assert result.attempt_count == 2
    assert len(transport.message_ids) == 2
    assert len(set(transport.message_ids)) == 1


def test_specialist_runtime_reloads_shared_provider_settings_without_restart(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_A2A_ENABLED": "true",
            "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
        }
    )
    runtime = create_default_a2a_runtime(settings, role=AgentRole.STOCK)

    initial = runtime.specialist_service.agent_runtime.resolve()
    runtime.persistence.runtime_settings.update(
        RuntimeSettingsOverrides(
            llm_provider="local-openai",
            llm_model="runtime-selected-model",
        ),
        base_settings=settings,
    )
    updated = runtime.specialist_service.agent_runtime.resolve()

    assert initial.provider_name == "offline-test"
    assert updated.provider_name == "local-openai"
    assert updated.model == "runtime-selected-model"


def test_specialist_runtime_resolves_hosted_credential_from_shared_keyring(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_A2A_ENABLED": "true",
            "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
        }
    )
    backend = MemoryKeyringBackend()
    credential_store = KeyringCredentialStore(backend=backend)
    credential_store.set("openai", "saved-provider-key")
    runtime = create_default_a2a_runtime(
        settings,
        role=AgentRole.STOCK,
        credential_store=credential_store,
    )
    runtime.persistence.runtime_settings.update(
        RuntimeSettingsOverrides(llm_provider="openai", llm_model="gpt-5-mini"),
        base_settings=settings,
    )

    selection = runtime.specialist_service.agent_runtime.resolve(require_research=True)

    assert selection.provider_name == "openai"
    assert selection.provider.api_key == "saved-provider-key"


def _runtime(
    tmp_path: Path,
    role: AgentRole,
) -> tuple[Settings, A2AResearchRuntime]:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_A2A_ENABLED": "true",
            "FRA_A2A_PUBLIC_BASE_URL": "http://specialist.test",
            "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
        }
    )
    persistence = create_persistence(settings)
    assert persistence.database is not None
    return settings, A2AResearchRuntime(
        task_store=SQLiteA2ATaskStore(persistence.database, owner=role.value),
        persistence=persistence,
        role=role,
        specialist_service=StubSpecialistService(),  # type: ignore[arg-type]
    )


class MemoryKeyringBackend:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


def _stock_request() -> DelegationRequest:
    return DelegationRequest(
        role=AgentRole.STOCK,
        run_id="orchestrator_run_test",
        step_id="stock_price_analysis",
        correlation_id="orchestrator_run_test",
        expected_kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        payload={"ticker": "TEST", "security_id": "security:test"},
    )


def _delegation_message(
    message_id: str,
    request: DelegationRequest,
) -> dict[str, object]:
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"data": request.to_dict(), "mediaType": "application/json"}],
        }
    }
