from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from financial_research_agent.orchestration import AgentRole
from financial_research_agent.persistence import SQLiteDatabase


@dataclass(frozen=True, slots=True)
class A2ADelegationRecord:
    id: str
    orchestrator_run_id: str
    correlation_id: str
    agent_role: AgentRole
    service_id: str
    status: str
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    remote_task_id: str | None = None
    error_code: str | None = None


class SQLiteA2ADelegationStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    async def save(self, record: A2ADelegationRecord) -> A2ADelegationRecord:
        return await asyncio.to_thread(self._save, record)

    async def list_for_run(self, run_id: str) -> tuple[A2ADelegationRecord, ...]:
        return await asyncio.to_thread(self._list_for_run, run_id)

    def _save(self, record: A2ADelegationRecord) -> A2ADelegationRecord:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO a2a_delegations(
                    id, orchestrator_run_id, correlation_id, agent_role, service_id,
                    remote_task_id, status, attempt_count, error_code, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    remote_task_id = excluded.remote_task_id,
                    status = excluded.status,
                    attempt_count = excluded.attempt_count,
                    error_code = excluded.error_code,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.orchestrator_run_id,
                    record.correlation_id,
                    record.agent_role.value,
                    record.service_id,
                    record.remote_task_id,
                    record.status,
                    record.attempt_count,
                    record.error_code,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def _list_for_run(self, run_id: str) -> tuple[A2ADelegationRecord, ...]:
        with self.database.read() as connection:
            rows = tuple(
                connection.execute(
                    """
                    SELECT *
                    FROM a2a_delegations
                    WHERE orchestrator_run_id = ?
                    ORDER BY created_at, id
                    """,
                    (run_id,),
                )
            )
        return tuple(
            A2ADelegationRecord(
                id=str(row["id"]),
                orchestrator_run_id=str(row["orchestrator_run_id"]),
                correlation_id=str(row["correlation_id"]),
                agent_role=AgentRole(str(row["agent_role"])),
                service_id=str(row["service_id"]),
                remote_task_id=row["remote_task_id"],
                status=str(row["status"]),
                attempt_count=int(row["attempt_count"]),
                error_code=row["error_code"],
                created_at=_datetime(row["created_at"]),
                updated_at=_datetime(row["updated_at"]),
            )
            for row in rows
        )


def delegation_record(
    *,
    delegation_id: str,
    run_id: str,
    correlation_id: str,
    role: AgentRole,
    service_id: str,
    status: str,
    attempt_count: int,
    remote_task_id: str | None = None,
    error_code: str | None = None,
) -> A2ADelegationRecord:
    now = datetime.now(UTC)
    return A2ADelegationRecord(
        id=delegation_id,
        orchestrator_run_id=run_id,
        correlation_id=correlation_id,
        agent_role=role,
        service_id=service_id,
        status=status,
        attempt_count=attempt_count,
        remote_task_id=remote_task_id,
        error_code=error_code,
        created_at=now,
        updated_at=now,
    )


def _datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
