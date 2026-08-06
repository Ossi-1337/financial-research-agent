from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Task, TaskState, TaskStatus
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

from financial_research_agent.a2a.specialists import SpecialistExecutionService
from financial_research_agent.a2a.store import SQLiteA2ATaskStore
from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import (
    AgentRole,
    DelegationRequest,
    delegation_request_from_dict,
)

MAX_DELEGATION_PAYLOAD_BYTES = 65_536
_ALLOWED_PAYLOAD_KEYS = {
    AgentRole.FINANCIAL_REPORT: {
        "company_id",
        "legal_name",
        "cik",
        "fiscal_years",
        "forms",
        "limit",
        "form_limits",
        "retrieval_query",
        "evidence_required",
    },
    AgentRole.STOCK: {
        "security_id",
        "ticker",
        "exchange_mic",
        "exchange_name",
        "currency",
        "benchmark_symbol",
        "outputsize",
        "evidence_required",
    },
    AgentRole.CONTEXT: {
        "query",
        "company_symbols",
        "source_items",
        "evidence_required",
        "web_research",
        "jurisdiction",
        "requires_official_source",
        "company_name",
    },
    AgentRole.SYNTHESIS: {"handoff_ids"},
}


class SpecialistAgentExecutor(AgentExecutor):
    def __init__(
        self,
        *,
        role: AgentRole,
        service: SpecialistExecutionService,
        task_store: SQLiteA2ATaskStore,
        redaction_policy: RedactionPolicy,
    ) -> None:
        self.role = role
        self.service = service
        self.task_store = task_store
        self.redaction_policy = redaction_policy
        self.artifact_redaction_policy = replace(
            redaction_policy,
            collection_preview_items=max(
                redaction_policy.collection_preview_items,
                128,
            ),
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = _required(context.task_id, "task_id")
        context_id = _required(context.context_id, "context_id")
        updater = TaskUpdater(event_queue, task_id, context_id)
        await event_queue.enqueue_event(_initial_task(context, updater))
        await self._wait_until_persisted(task_id)
        try:
            request = _delegation_request(context)
            if request.role != self.role:
                raise ValueError("delegation role does not match this specialist")
        except ValueError as exc:
            await updater.reject(_agent_message(updater, str(exc)))
            return
        await updater.start_work(_agent_message(updater, f"{self.role.value} started."))
        await self.task_store.append_event(
            task_id,
            "working",
            {
                "role": self.role.value,
                "correlation_id": request.correlation_id,
                "run_id": request.run_id,
            },
        )
        try:
            handoff = await self.service.execute(request)
        except Exception:
            await updater.failed(_agent_message(updater, "Specialist task failed safely."))
            await self.task_store.append_event(
                task_id,
                "failed",
                {"error_code": "specialist_execution_failed"},
            )
            return
        safe_payload = self.artifact_redaction_policy.redact(handoff.to_dict())
        await updater.add_artifact(
            [_data_part(safe_payload)],
            artifact_id=f"specialist-{handoff.id}",
            name=self.role.value,
            metadata={
                "kind": "specialist_handoff",
                "role": self.role.value,
                "correlation_id": request.correlation_id,
            },
            last_chunk=True,
        )
        await self.task_store.append_event(task_id, "handoff", safe_payload)
        await updater.complete(_agent_message(updater, f"{self.role.value} completed."))
        await self.task_store.append_event(
            task_id,
            "completed",
            {"handoff_id": handoff.id, "handoff_status": handoff.status.value},
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = _required(context.task_id, "task_id")
        updater = TaskUpdater(
            event_queue,
            task_id,
            _required(context.context_id, "context_id"),
        )
        await updater.cancel(_agent_message(updater, "Specialist task cancelled."))
        await self.task_store.append_event(task_id, "cancelled", {"error_code": "cancelled"})

    async def _wait_until_persisted(self, task_id: str) -> None:
        for _attempt in range(100):
            if await self.task_store.exists(task_id):
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("A2A specialist task was not persisted")


def _delegation_request(context: RequestContext) -> DelegationRequest:
    message = context.message
    if message is None or len(message.parts) != 1:
        raise ValueError("one application/json DataPart is required")
    part = message.parts[0]
    if part.WhichOneof("content") != "data" or part.media_type != "application/json":
        raise ValueError("one application/json DataPart is required")
    data = MessageToDict(part).get("data")
    if not isinstance(data, dict):
        raise ValueError("delegation payload must be an object")
    if len(json.dumps(data, separators=(",", ":")).encode("utf-8")) > MAX_DELEGATION_PAYLOAD_BYTES:
        raise ValueError("delegation payload is too large")
    request = delegation_request_from_dict(data)
    allowed = _ALLOWED_PAYLOAD_KEYS.get(request.role)
    if allowed is None or set(request.payload) - allowed:
        raise ValueError("delegation payload contains unsupported fields")
    return request


def _initial_task(context: RequestContext, updater: TaskUpdater) -> Task:
    timestamp = Timestamp()
    timestamp.FromDatetime(datetime.now(UTC))
    return Task(
        id=updater.task_id,
        context_id=updater.context_id,
        status=TaskStatus(
            state=TaskState.TASK_STATE_SUBMITTED,
            message=_agent_message(updater, "Specialist request received."),
            timestamp=timestamp,
        ),
        history=[context.message] if context.message is not None else [],
    )


def _data_part(payload: object) -> Part:
    return ParseDict(
        {"data": payload, "mediaType": "application/json"},
        Part(),
    )


def _agent_message(updater: TaskUpdater, text: str) -> Message:
    return updater.new_agent_message([Part(text=text)])


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value
