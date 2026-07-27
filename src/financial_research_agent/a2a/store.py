from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import (
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    Task,
    TaskState,
)
from a2a.utils.errors import InvalidParamsError
from google.protobuf.timestamp_pb2 import Timestamp

from financial_research_agent.persistence import SQLiteDatabase

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
ACTIVE_TASK_STATES = (
    TaskState.TASK_STATE_SUBMITTED,
    TaskState.TASK_STATE_WORKING,
)
TERMINAL_TASK_STATES = (
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
)


@dataclass(frozen=True, slots=True)
class A2ATaskEventRecord:
    task_id: str
    sequence: int
    event_type: str
    created_at: datetime
    payload: dict[str, object]


class SQLiteA2ATaskStore(TaskStore):
    def __init__(self, database: SQLiteDatabase, *, owner: str = "company-research") -> None:
        self.database = database
        self.owner = _required_text("owner", owner)

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await asyncio.to_thread(self._save, task, self.owner)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        return await asyncio.to_thread(self._get, task_id, self.owner)

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        return await asyncio.to_thread(self._list, params, self.owner)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await asyncio.to_thread(self._delete, task_id, self.owner)

    async def find_by_message_id(self, message_id: str) -> Task | None:
        return await asyncio.to_thread(self._find_by_message_id, message_id, self.owner)

    async def exists(self, task_id: str) -> bool:
        return await asyncio.to_thread(self._exists, task_id)

    async def bind_execution(
        self,
        task_id: str,
        *,
        background_job_id: str,
        orchestrator_run_id: str,
    ) -> None:
        await asyncio.to_thread(
            self._bind_execution,
            task_id,
            background_job_id,
            orchestrator_run_id,
        )

    async def append_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> A2ATaskEventRecord:
        return await asyncio.to_thread(self._append_event, task_id, event_type, payload)

    async def execution_ids(self, task_id: str) -> tuple[str | None, str | None]:
        return await asyncio.to_thread(self._execution_ids, task_id)

    async def events(self, task_id: str) -> tuple[A2ATaskEventRecord, ...]:
        return await asyncio.to_thread(self._events, task_id)

    async def reconcile_restarted_tasks(self) -> int:
        return await asyncio.to_thread(self._reconcile_restarted_tasks, self.owner)

    def _save(self, task: Task, owner: str) -> None:
        if not task.id or not task.context_id:
            raise ValueError("A2A task id and context id are required")
        now = datetime.now(UTC).isoformat()
        message_id = _initial_message_id(task)
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT initial_message_id, created_at, state FROM a2a_tasks WHERE id = ?",
                (task.id,),
            ).fetchone()
            if existing is not None:
                message_id = str(existing["initial_message_id"])
                created_at = str(existing["created_at"])
                existing_state = int(existing["state"])
                if existing_state in TERMINAL_TASK_STATES and task.status.state != existing_state:
                    return
            else:
                if message_id is None:
                    raise ValueError("A2A task must preserve its initial message id")
                created_at = now
            connection.execute(
                """
                INSERT INTO a2a_tasks(
                    id, owner, context_id, state, initial_message_id,
                    created_at, updated_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    owner = excluded.owner,
                    context_id = excluded.context_id,
                    state = excluded.state,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    task.id,
                    owner,
                    task.context_id,
                    int(task.status.state),
                    message_id,
                    created_at,
                    now,
                    task.SerializeToString(),
                ),
            )

    def _get(self, task_id: str, owner: str) -> Task | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload FROM a2a_tasks WHERE id = ? AND owner = ?",
                (_required_text("task_id", task_id), owner),
            ).fetchone()
        return _task_from_payload(row["payload"]) if row is not None else None

    def _list(self, params: ListTasksRequest, owner: str) -> ListTasksResponse:
        query = "SELECT id, payload FROM a2a_tasks WHERE owner = ?"
        arguments: list[object] = [owner]
        if params.context_id:
            query += " AND context_id = ?"
            arguments.append(params.context_id)
        if params.status:
            query += " AND state = ?"
            arguments.append(int(params.status))
        query += " ORDER BY updated_at DESC, id DESC"
        with self.database.read() as connection:
            tasks = [
                _task_from_payload(row["payload"])
                for row in connection.execute(query, tuple(arguments))
            ]
        if params.HasField("status_timestamp_after"):
            threshold = params.status_timestamp_after.ToDatetime(tzinfo=UTC)
            tasks = [
                task
                for task in tasks
                if task.status.HasField("timestamp")
                and task.status.timestamp.ToDatetime(tzinfo=UTC) > threshold
            ]
        total_size = len(tasks)
        start = _page_start(tasks, params.page_token)
        page_size = min(params.page_size or DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE)
        selected = tasks[start : start + page_size]
        normalized = [
            _task_for_list(
                task,
                history_length=params.history_length,
                include_artifacts=params.include_artifacts,
            )
            for task in selected
        ]
        next_index = start + len(selected)
        next_token = _encode_page_token(tasks[next_index].id) if next_index < total_size else ""
        return ListTasksResponse(
            tasks=normalized,
            next_page_token=next_token,
            page_size=page_size,
            total_size=total_size,
        )

    def _delete(self, task_id: str, owner: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM a2a_tasks WHERE id = ? AND owner = ?",
                (_required_text("task_id", task_id), owner),
            )

    def _find_by_message_id(self, message_id: str, owner: str) -> Task | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload FROM a2a_tasks WHERE initial_message_id = ? AND owner = ?",
                (_required_text("message_id", message_id), owner),
            ).fetchone()
        return _task_from_payload(row["payload"]) if row is not None else None

    def _exists(self, task_id: str) -> bool:
        with self.database.read() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM a2a_tasks WHERE id = ?",
                    (_required_text("task_id", task_id),),
                ).fetchone()
                is not None
            )

    def _bind_execution(
        self,
        task_id: str,
        background_job_id: str,
        orchestrator_run_id: str,
    ) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE a2a_tasks
                SET background_job_id = ?, orchestrator_run_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _required_text("background_job_id", background_job_id),
                    _required_text("orchestrator_run_id", orchestrator_run_id),
                    datetime.now(UTC).isoformat(),
                    _required_text("task_id", task_id),
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(task_id)

    def _append_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> A2ATaskEventRecord:
        created_at = datetime.now(UTC)
        safe_payload = json.loads(json.dumps(payload))
        with self.database.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM a2a_tasks WHERE id = ?",
                    (_required_text("task_id", task_id),),
                ).fetchone()
                is None
            ):
                raise KeyError(task_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 AS sequence "
                "FROM a2a_task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            sequence = int(row["sequence"])
            connection.execute(
                """
                INSERT INTO a2a_task_events(
                    task_id, sequence, event_type, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    sequence,
                    _required_text("event_type", event_type),
                    created_at.isoformat(),
                    json.dumps(safe_payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        return A2ATaskEventRecord(task_id, sequence, event_type, created_at, safe_payload)

    def _execution_ids(self, task_id: str) -> tuple[str | None, str | None]:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT background_job_id, orchestrator_run_id
                FROM a2a_tasks
                WHERE id = ?
                """,
                (_required_text("task_id", task_id),),
            ).fetchone()
        if row is None:
            return None, None
        return row["background_job_id"], row["orchestrator_run_id"]

    def _events(self, task_id: str) -> tuple[A2ATaskEventRecord, ...]:
        with self.database.read() as connection:
            rows = tuple(
                connection.execute(
                    """
                    SELECT sequence, event_type, created_at, payload_json
                    FROM a2a_task_events
                    WHERE task_id = ?
                    ORDER BY sequence
                    """,
                    (_required_text("task_id", task_id),),
                )
            )
        return tuple(
            A2ATaskEventRecord(
                task_id=task_id,
                sequence=int(row["sequence"]),
                event_type=str(row["event_type"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def _reconcile_restarted_tasks(self, owner: str) -> int:
        with self.database.transaction() as connection:
            rows = tuple(
                connection.execute(
                    "SELECT id, payload FROM a2a_tasks WHERE owner = ? AND state IN (?, ?)",
                    (owner, *ACTIVE_TASK_STATES),
                )
            )
            for row in rows:
                task = _task_from_payload(row["payload"])
                timestamp = Timestamp()
                timestamp.FromDatetime(datetime.now(UTC))
                task.status.state = TaskState.TASK_STATE_FAILED
                task.status.timestamp.CopyFrom(timestamp)
                task.status.message.CopyFrom(
                    Message(
                        message_id=f"restart-{task.id}",
                        task_id=task.id,
                        context_id=task.context_id,
                        role=Role.ROLE_AGENT,
                        parts=[
                            Part(
                                text=(
                                    "Research task stopped because the local A2A service restarted."
                                )
                            )
                        ],
                    )
                )
                connection.execute(
                    "UPDATE a2a_tasks SET state = ?, updated_at = ?, payload = ? WHERE id = ?",
                    (
                        TaskState.TASK_STATE_FAILED,
                        datetime.now(UTC).isoformat(),
                        task.SerializeToString(),
                        task.id,
                    ),
                )
                sequence_row = connection.execute(
                    "SELECT COALESCE(MAX(sequence), -1) + 1 AS sequence "
                    "FROM a2a_task_events WHERE task_id = ?",
                    (task.id,),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO a2a_task_events(
                        task_id, sequence, event_type, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task.id,
                        int(sequence_row["sequence"]),
                        "process_restarted",
                        datetime.now(UTC).isoformat(),
                        '{"error_code":"process_restarted"}',
                    ),
                )
        return len(rows)


def _initial_message_id(task: Task) -> str | None:
    return task.history[0].message_id if task.history and task.history[0].message_id else None


def _task_from_payload(payload: bytes) -> Task:
    task = Task()
    task.ParseFromString(payload)
    return task


def _task_for_list(task: Task, *, history_length: int, include_artifacts: bool) -> Task:
    value = Task()
    value.CopyFrom(task)
    if history_length > 0 and len(value.history) > history_length:
        del value.history[:-history_length]
    elif history_length == 0:
        del value.history[:]
    if not include_artifacts:
        del value.artifacts[:]
    return value


def _page_start(tasks: list[Task], token: str) -> int:
    if not token:
        return 0
    task_id = _decode_page_token(token)
    for index, task in enumerate(tasks):
        if task.id == task_id:
            return index + 1
    raise InvalidParamsError(f"Invalid page token: {token}")


def _encode_page_token(task_id: str) -> str:
    return base64.urlsafe_b64encode(task_id.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_page_token(token: str) -> str:
    try:
        padding = "=" * (-len(token) % 4)
        return base64.urlsafe_b64decode(f"{token}{padding}").decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidParamsError(f"Invalid page token: {token}") from exc


def _required_text(name: str, value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text
