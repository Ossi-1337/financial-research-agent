from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Task, TaskState, TaskStatus
from google.protobuf.json_format import ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

from financial_research_agent.a2a.store import SQLiteA2ATaskStore
from financial_research_agent.background import (
    BackgroundResearchRunner,
    BackgroundResearchStatus,
)
from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import (
    AgentHandoff,
    OrchestratedResearchRun,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    ResearchOrchestrator,
)
from financial_research_agent.settings import A2ASettings

POLL_SECONDS = 0.05


class CompanyResearchAgentExecutor(AgentExecutor):
    def __init__(
        self,
        *,
        settings: A2ASettings,
        orchestrator: ResearchOrchestrator,
        background_runner: BackgroundResearchRunner,
        task_store: SQLiteA2ATaskStore,
        orchestrator_run_store: object,
        redaction_policy: RedactionPolicy,
    ) -> None:
        self.settings = settings
        self.orchestrator = orchestrator
        self.background_runner = background_runner
        self.task_store = task_store
        self.orchestrator_run_store = orchestrator_run_store
        self.redaction_policy = redaction_policy

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = _required(context.task_id, "task_id")
        context_id = _required(context.context_id, "context_id")
        updater = TaskUpdater(event_queue, task_id, context_id)
        await event_queue.enqueue_event(_initial_task(context, updater))
        await self._wait_until_persisted(task_id)
        try:
            query = _validate_request(context, self.settings.max_input_chars)
        except ValueError as exc:
            await updater.reject(_agent_message(updater, str(exc)))
            return

        stats = await self.background_runner.stats()
        if int(stats["queued_count"]) >= self.settings.max_queued_tasks:
            await updater.reject(
                _agent_message(updater, "Local A2A research queue is full. Retry later.")
            )
            return

        await self.task_store.append_event(
            task_id,
            "submitted",
            {"query_chars": len(query), "skill": "company_research"},
        )

        progress_queue: asyncio.Queue[OrchestratedResearchRun] = asyncio.Queue()

        async def run_research(request: OrchestratorResearchInput) -> OrchestratedResearchRun:
            return await self.orchestrator.run(
                request,
                progress_observer=progress_queue.put_nowait,
            )

        job = await self.background_runner.submit(
            OrchestratorResearchInput(query=query),
            run=run_research,
            metadata={"a2a_task_id": task_id, "a2a_context_id": context_id},
        )
        await self.task_store.bind_execution(
            task_id,
            background_job_id=job.id,
            orchestrator_run_id=job.orchestrator_run_id,
        )
        await updater.start_work(_agent_message(updater, "Source-backed research started."))
        await self.task_store.append_event(
            task_id,
            "working",
            {
                "background_job_id": job.id,
                "orchestrator_run_id": job.orchestrator_run_id,
            },
        )

        emitted_handoffs: set[str] = set()
        while True:
            await self._emit_progress(progress_queue, updater, task_id, emitted_handoffs)
            current = await self.background_runner.get(job.id)
            if current is None:
                await self._fail(updater, task_id, "background_job_missing")
                return
            if current.status in {
                BackgroundResearchStatus.SUCCEEDED,
                BackgroundResearchStatus.FAILED,
                BackgroundResearchStatus.CANCELLED,
            }:
                break
            await asyncio.sleep(POLL_SECONDS)

        await self._emit_progress(progress_queue, updater, task_id, emitted_handoffs)
        if current.status == BackgroundResearchStatus.CANCELLED:
            await updater.cancel(_agent_message(updater, "Research task cancelled."))
            await self.task_store.append_event(task_id, "cancelled", {"error_code": "cancelled"})
            return
        if current.status == BackgroundResearchStatus.FAILED:
            await self._fail(updater, task_id, current.error_code or "research_failed")
            return

        run = self.orchestrator_run_store.get(job.orchestrator_run_id)
        if run is None:
            await self._fail(updater, task_id, "orchestrator_run_missing")
            return
        await self._emit_final_artifact(updater, task_id, run)
        if run.status == OrchestratorRunStatus.FAILED:
            await self._fail(updater, task_id, "orchestrator_failed")
            return
        await updater.complete(
            _agent_message(
                updater,
                run.synthesis_summary or "Research completed with inspectable limitations.",
            )
        )
        await self.task_store.append_event(
            task_id,
            "completed",
            {"run_id": run.id, "run_status": run.status.value},
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = _required(context.task_id, "task_id")
        context_id = _required(context.context_id, "context_id")
        updater = TaskUpdater(event_queue, task_id, context_id)
        background_job_id, _run_id = await self.task_store.execution_ids(task_id)
        if background_job_id is not None:
            await self.background_runner.cancel(background_job_id)
        message = _agent_message(updater, "Research task cancelled.")
        await updater.cancel(message)
        task = await self.task_store.get(task_id, context.call_context)
        if task is not None:
            timestamp = Timestamp()
            timestamp.FromDatetime(datetime.now(UTC))
            task.status.state = TaskState.TASK_STATE_CANCELED
            task.status.timestamp.CopyFrom(timestamp)
            task.status.message.CopyFrom(message)
            await self.task_store.save(task, context.call_context)
        await self.task_store.append_event(task_id, "cancelled", {"error_code": "cancelled"})

    async def _wait_until_persisted(self, task_id: str) -> None:
        for _attempt in range(100):
            if await self.task_store.exists(task_id):
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("A2A task was not persisted")

    async def _emit_progress(
        self,
        queue: asyncio.Queue[OrchestratedResearchRun],
        updater: TaskUpdater,
        task_id: str,
        emitted_handoffs: set[str],
    ) -> None:
        while not queue.empty():
            run = queue.get_nowait()
            for handoff in run.handoffs:
                if handoff.id in emitted_handoffs:
                    continue
                emitted_handoffs.add(handoff.id)
                payload = _handoff_payload(handoff, self.redaction_policy)
                await updater.add_artifact(
                    [_data_part(payload)],
                    artifact_id=f"handoff-{handoff.id}",
                    name=handoff.kind.value,
                    metadata={"kind": "specialist_handoff", "step_id": handoff.step_id},
                    last_chunk=True,
                )
                await self.task_store.append_event(task_id, "handoff", payload)

    async def _emit_final_artifact(
        self,
        updater: TaskUpdater,
        task_id: str,
        run: OrchestratedResearchRun,
    ) -> None:
        payload = _final_payload(run, self.redaction_policy)
        await updater.add_artifact(
            [_data_part(payload)],
            artifact_id=f"research-report-{run.id}",
            name="source-backed-research-report",
            metadata={"kind": "deterministic_synthesis", "run_id": run.id},
            last_chunk=True,
        )
        await self.task_store.append_event(task_id, "final_artifact", payload)

    async def _fail(self, updater: TaskUpdater, task_id: str, error_code: str) -> None:
        safe_code = _safe_error_code(error_code)
        await updater.failed(
            _agent_message(
                updater,
                f"Research task failed safely. Error code: {safe_code}.",
            )
        )
        await self.task_store.append_event(task_id, "failed", {"error_code": safe_code})


def _validate_request(context: RequestContext, max_input_chars: int) -> str:
    message = context.message
    if message is None:
        raise ValueError("One text message is required.")
    if context.call_context.state.get("fra_original_task_id"):
        raise ValueError("Multi-turn A2A tasks are not supported.")
    if message.metadata or message.extensions or message.reference_task_ids:
        raise ValueError("Message metadata, extensions, and task references are not supported.")
    if context.metadata:
        raise ValueError("Request metadata is not supported.")
    if len(message.parts) != 1 or message.parts[0].WhichOneof("content") != "text":
        raise ValueError("Exactly one text part is required.")
    query = message.parts[0].text.strip()
    if not query:
        raise ValueError("Research query is required.")
    if len(query) > max_input_chars:
        raise ValueError(f"Research query exceeds {max_input_chars} characters.")
    return query


def _handoff_payload(
    handoff: AgentHandoff,
    redaction_policy: RedactionPolicy,
) -> dict[str, object]:
    return {
        "handoff_id": handoff.id,
        "step_id": handoff.step_id,
        "kind": handoff.kind.value,
        "status": handoff.status.value,
        "confidence": handoff.confidence.value,
        "evidence_ids": list(handoff.evidence_ids),
        "warnings": list(handoff.warnings),
        "limitations": list(handoff.limitations),
        "output": _plain_json(redaction_policy.redact(dict(handoff.output))),
    }


def _final_payload(
    run: OrchestratedResearchRun,
    redaction_policy: RedactionPolicy,
) -> dict[str, object]:
    synthesis = next(
        (handoff for handoff in reversed(run.handoffs) if handoff.step_id == "synthesis"),
        None,
    )
    return {
        "run_id": run.id,
        "status": run.status.value,
        "company": _plain_json(redaction_policy.redact(run.selected_company)),
        "security": _plain_json(redaction_policy.redact(run.selected_security)),
        "synthesis": (
            _plain_json(redaction_policy.redact(dict(synthesis.output)))
            if synthesis is not None
            else None
        ),
        "evidence_ids": list(
            dict.fromkeys(
                evidence_id for handoff in run.handoffs for evidence_id in handoff.evidence_ids
            )
        ),
        "warnings": list(run.warnings),
        "limitations": list(run.limitations),
        "no_recommendation_notice": run.no_recommendation_notice,
    }


def _data_part(payload: Mapping[str, object]) -> Part:
    return ParseDict(
        {"data": _plain_json(payload), "mediaType": "application/json"},
        Part(),
    )


def _agent_message(updater: TaskUpdater, text: str) -> Message:
    return updater.new_agent_message([Part(text=text)])


def _initial_task(context: RequestContext, updater: TaskUpdater) -> Task:
    timestamp = Timestamp()
    timestamp.FromDatetime(datetime.now(UTC))
    status = TaskStatus(
        state=TaskState.TASK_STATE_SUBMITTED,
        message=_agent_message(updater, "Research request received."),
        timestamp=timestamp,
    )
    history = [context.message] if context.message is not None else []
    return Task(
        id=updater.task_id,
        context_id=updater.context_id,
        status=status,
        history=history,
    )


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _safe_error_code(value: str) -> str:
    normalized = "".join(
        character for character in value if character.isalnum() or character in "_-"
    )
    return normalized[:80] or "research_failed"


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value
