from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from a2a.server.context import ServerCallContext
from a2a.types import ListTasksRequest, Task
from google.protobuf.json_format import ParseDict

from financial_research_agent.a2a import (
    SQLiteA2ADelegationStore,
    SQLiteA2ATaskStore,
)
from financial_research_agent.a2a.delegations import delegation_record
from financial_research_agent.orchestration import (
    AgentRole,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorRunStatus,
    default_orchestrator_plan,
)
from financial_research_agent.persistence import create_persistence
from financial_research_agent.settings import Settings


def test_a2a_tasks_are_listed_only_for_the_service_owner(tmp_path: Path) -> None:
    persistence = create_persistence(_settings(tmp_path))
    assert persistence.database is not None
    financial = SQLiteA2ATaskStore(persistence.database, owner="financial-report")
    stock = SQLiteA2ATaskStore(persistence.database, owner="stock")
    context = ServerCallContext(state={})

    async def exercise() -> None:
        await financial.save(_task("task-financial", "message-financial"), context)
        await stock.save(_task("task-stock", "message-stock"), context)
        financial_tasks = await financial.list(ListTasksRequest(), context)
        stock_tasks = await stock.list(ListTasksRequest(), context)

        assert [task.id for task in financial_tasks.tasks] == ["task-financial"]
        assert [task.id for task in stock_tasks.tasks] == ["task-stock"]

    asyncio.run(exercise())


def test_a2a_events_remain_ordered_under_concurrent_appends(tmp_path: Path) -> None:
    persistence = create_persistence(_settings(tmp_path))
    assert persistence.database is not None
    store = SQLiteA2ATaskStore(persistence.database, owner="context")
    context = ServerCallContext(state={})

    async def exercise() -> None:
        await store.save(_task("task-events", "message-events"), context)
        await asyncio.gather(
            *(
                store.append_event("task-events", "progress", {"index": index})
                for index in range(20)
            )
        )
        events = await store.events("task-events")
        assert [event.sequence for event in events] == list(range(20))
        assert {int(event.payload["index"]) for event in events} == set(range(20))

    asyncio.run(exercise())


def test_delegation_records_round_trip_with_orchestrator_relation(tmp_path: Path) -> None:
    persistence = create_persistence(_settings(tmp_path))
    assert persistence.database is not None
    now = datetime(2026, 7, 27, 12, tzinfo=UTC)
    persistence.orchestrator_runs.save(
        OrchestratedResearchRun(
            id="run-delegation",
            query="Research test company",
            status=OrchestratorRunStatus.RUNNING,
            created_at=now,
            updated_at=now,
            execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
            plan=default_orchestrator_plan(),
        )
    )
    store = SQLiteA2ADelegationStore(persistence.database)
    record = delegation_record(
        delegation_id="delegation-1",
        run_id="run-delegation",
        correlation_id="run-delegation",
        role=AgentRole.STOCK,
        service_id="stock",
        status="succeeded",
        attempt_count=2,
        remote_task_id="remote-task-1",
    )

    asyncio.run(store.save(record))
    loaded = asyncio.run(store.list_for_run("run-delegation"))

    assert len(loaded) == 1
    assert loaded[0].agent_role == AgentRole.STOCK
    assert loaded[0].remote_task_id == "remote-task-1"
    assert loaded[0].attempt_count == 2


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_SEC_USER_AGENT": "financial-research-agent tests tests@example.com",
        }
    )


def _task(task_id: str, message_id: str) -> Task:
    return ParseDict(
        {
            "id": task_id,
            "contextId": f"context-{task_id}",
            "status": {"state": "TASK_STATE_SUBMITTED"},
            "history": [
                {
                    "messageId": message_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": "TEST TOOL OUTPUT"}],
                }
            ],
        },
        Task(),
    )
