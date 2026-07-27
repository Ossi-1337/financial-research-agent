from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from a2a.types import AgentCard, Task, TaskState
from fastapi.testclient import TestClient
from google.protobuf.json_format import ParseDict

from financial_research_agent.a2a import (
    A2AResearchRuntime,
    SQLiteA2ATaskStore,
    create_a2a_app,
)
from financial_research_agent.background import BackgroundResearchRunner
from financial_research_agent.orchestration import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.persistence import create_persistence
from financial_research_agent.settings import Settings

NOW = datetime(2026, 7, 27, 12, tzinfo=UTC)
A2A_HEADERS = {"A2A-Version": "1.0"}


class StubOrchestrator:
    def __init__(self, run_store: object) -> None:
        self.run_store = run_store

    async def run(self, request, *, progress_observer=None) -> OrchestratedResearchRun:
        handoff = AgentHandoff(
            id=f"handoff:{request.run_id}:synthesis",
            step_id="synthesis",
            kind=OrchestratorStepKind.SYNTHESIS,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
            output={
                "report": {
                    "summary": "Fixture-backed deterministic research.",
                    "source_marker": "[S1]",
                }
            },
            evidence_ids=("evidence:test:1",),
            confidence=HandoffConfidence.MEDIUM,
        )
        run = OrchestratedResearchRun(
            id=request.run_id,
            query=request.query,
            status=OrchestratorRunStatus.COMPLETE,
            created_at=NOW,
            updated_at=NOW,
            execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
            plan=default_orchestrator_plan(),
            handoffs=(handoff,),
            selected_company={"legal_name": "TEST TOOL OUTPUT COMPANY"},
            selected_security={"ticker": "TEST"},
            synthesis_summary="Fixture-backed deterministic research.",
            warnings=("TEST FIXTURE OUTPUT ONLY.",),
            limitations=("No live providers used.",),
        )
        self.run_store.save(run)
        if progress_observer is not None:
            progress_observer(run)
        return run


class SlowOrchestrator:
    async def run(self, request, *, progress_observer=None) -> OrchestratedResearchRun:
        await asyncio.sleep(30)
        raise AssertionError("cancelled research must not complete")


def test_agent_card_and_supported_routes_use_a2a_1_0(tmp_path: Path) -> None:
    settings, runtime = _runtime(tmp_path)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        response = client.get("/.well-known/agent-card.json")
        card = ParseDict(response.json(), AgentCard())

        assert response.status_code == 200
        assert response.headers["etag"]
        assert response.headers["cache-control"] == "public, max-age=300"
        assert card.supported_interfaces[0].protocol_version == "1.0"
        assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"
        assert card.skills[0].id == "company_research"
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is False
        assert client.get("/extendedAgentCard", headers=A2A_HEADERS).status_code == 404
        assert client.get("/.well-known/agent.json").status_code == 404
        assert client.get("/tasks", headers=A2A_HEADERS).status_code == 200


def test_send_get_list_and_idempotent_message_id(tmp_path: Path) -> None:
    settings, runtime = _runtime(tmp_path)
    request = _message_request("message-1", "Research Tesla")

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        first = client.post("/message:send", headers=A2A_HEADERS, json=request)
        second = client.post("/message:send", headers=A2A_HEADERS, json=request)

        assert first.status_code == 200
        assert second.status_code == 200
        first_task = first.json()["task"]
        assert second.json()["task"]["id"] == first_task["id"]
        assert first_task["status"]["state"] == "TASK_STATE_COMPLETED"
        task_id = first_task["id"]
        fetched = client.get(f"/tasks/{task_id}", headers=A2A_HEADERS)
        listed = client.get("/tasks", headers=A2A_HEADERS)

        assert fetched.json()["id"] == task_id
        assert [task["id"] for task in listed.json()["tasks"]] == [task_id]
        assert first_task["artifacts"][-1]["artifactId"].startswith("research-report-")
        assert "no_recommendation_notice" in json.dumps(first_task)

    events = asyncio.run(runtime.task_store.events(task_id))
    assert [event.sequence for event in events] == list(range(len(events)))
    assert events[-1].event_type == "completed"


def test_stream_returns_sse_and_persists_completed_task(tmp_path: Path) -> None:
    settings, runtime = _runtime(tmp_path)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        response = client.post(
            "/message:stream",
            headers=A2A_HEADERS,
            json=_message_request("stream-1", "Research Novo Nordisk"),
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "TASK_STATE_SUBMITTED" in response.text
        assert "TASK_STATE_COMPLETED" in response.text


def test_terminal_task_can_be_resubscribed_from_persisted_snapshot(tmp_path: Path) -> None:
    settings, runtime = _runtime(tmp_path)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        sent = client.post(
            "/message:send",
            headers=A2A_HEADERS,
            json=_message_request("subscribe-1", "Research Tesla"),
        )
        task_id = sent.json()["task"]["id"]
        response = client.post(
            f"/tasks/{task_id}:subscribe",
            headers=A2A_HEADERS,
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "TASK_STATE_COMPLETED" in response.text


def test_active_task_can_be_cancelled(tmp_path: Path) -> None:
    settings, runtime = _runtime(tmp_path, orchestrator=SlowOrchestrator())
    request = _message_request("cancel-1", "Research Tesla")
    request["configuration"] = {"returnImmediately": True}

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        sent = client.post("/message:send", headers=A2A_HEADERS, json=request)
        task_id = sent.json()["task"]["id"]
        cancelled = client.post(
            f"/tasks/{task_id}:cancel",
            headers=A2A_HEADERS,
        )

        assert sent.status_code == 200
        assert cancelled.status_code == 200
        assert cancelled.json()["status"]["state"] == "TASK_STATE_CANCELED"
        time.sleep(0.25)
        persisted = client.get(f"/tasks/{task_id}", headers=A2A_HEADERS)
        assert persisted.json()["status"]["state"] == "TASK_STATE_CANCELED"
        background_job_id, _run_id = asyncio.run(runtime.task_store.execution_ids(task_id))
        assert background_job_id is not None
        job = asyncio.run(runtime.background_runner.get(background_job_id))
        assert job is not None
        assert job.status.value == "cancelled"


@pytest.mark.parametrize(
    ("request_payload", "expected_state"),
    [
        (
            {
                "message": {
                    "messageId": "file-1",
                    "role": "ROLE_USER",
                    "parts": [{"raw": "bm90LWFsbG93ZWQ=", "mediaType": "text/plain"}],
                }
            },
            "TASK_STATE_REJECTED",
        ),
        (
            {
                "message": {
                    "messageId": "empty-1",
                    "role": "ROLE_USER",
                    "parts": [{"text": "   "}],
                }
            },
            "TASK_STATE_REJECTED",
        ),
    ],
)
def test_invalid_content_is_rejected(
    tmp_path: Path,
    request_payload: dict[str, object],
    expected_state: str,
) -> None:
    settings, runtime = _runtime(tmp_path)

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        response = client.post(
            "/message:send",
            headers=A2A_HEADERS,
            json=request_payload,
        )

        assert response.status_code == 200
        assert response.json()["task"]["status"]["state"] == expected_state


def test_remote_mode_requires_bearer_and_card_contains_no_key(tmp_path: Path) -> None:
    settings, runtime = _runtime(
        tmp_path,
        FRA_A2A_LOCAL_ONLY="false",
        FRA_A2A_API_KEY="a2a-secret",
        FRA_A2A_PUBLIC_BASE_URL="https://research.example.test/a2a",
    )

    with TestClient(create_a2a_app(settings=settings, runtime=runtime)) as client:
        card = client.get("/.well-known/agent-card.json")
        denied = client.get("/tasks", headers=A2A_HEADERS)
        allowed = client.get(
            "/tasks",
            headers={**A2A_HEADERS, "Authorization": "Bearer a2a-secret"},
        )

        assert denied.status_code == 401
        assert allowed.status_code == 200
        assert "a2a-secret" not in card.text
        assert "securitySchemes" in card.json()


def test_unfinished_task_recovery_marks_task_failed(tmp_path: Path) -> None:
    _settings, runtime = _runtime(tmp_path)
    task_id = "a2a_task_recovery"
    task = ParseDict(
        {
            "id": task_id,
            "contextId": "a2a_context_recovery",
            "status": {"state": "TASK_STATE_WORKING"},
            "history": [
                {
                    "messageId": "recovery-1",
                    "role": "ROLE_USER",
                    "parts": [{"text": "Research a company"}],
                }
            ],
        },
        Task(),
    )
    asyncio.run(runtime.task_store.save(task, _local_call_context()))

    assert asyncio.run(runtime.task_store.reconcile_restarted_tasks()) == 1
    recovered = asyncio.run(runtime.task_store.find_by_message_id("recovery-1"))
    assert recovered is not None
    assert recovered.status.state == TaskState.TASK_STATE_FAILED
    recovery_events = asyncio.run(runtime.task_store.events(task_id))
    assert recovery_events[-1].payload["error_code"] == "process_restarted"
    assert task_id == recovered.id


def _runtime(
    tmp_path: Path,
    orchestrator: object | None = None,
    **overrides: str,
) -> tuple[Settings, A2AResearchRuntime]:
    values = {
        "FRA_HOME": str(tmp_path),
        "FRA_A2A_ENABLED": "true",
        "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
        **overrides,
    }
    settings = Settings.from_env(values)
    persistence = create_persistence(settings)
    assert persistence.database is not None
    orchestrator = orchestrator or StubOrchestrator(persistence.orchestrator_runs)
    return settings, A2AResearchRuntime(
        orchestrator=orchestrator,
        background_runner=BackgroundResearchRunner(
            max_concurrent_runs=1,
            job_store=persistence.background_jobs,
        ),
        task_store=SQLiteA2ATaskStore(persistence.database),
        orchestrator_run_store=persistence.orchestrator_runs,
        persistence=persistence,
    )


def _message_request(message_id: str, query: str) -> dict[str, object]:
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"text": query}],
        }
    }


def _local_call_context():
    from a2a.server.context import ServerCallContext

    return ServerCallContext(state={})
