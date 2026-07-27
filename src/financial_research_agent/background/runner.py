from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, Self
from uuid import uuid4

from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorResearchInput,
)


class BackgroundResearchStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundQueueFullError(RuntimeError):
    """Raised when an atomic background queue admission check fails."""


@dataclass(frozen=True, slots=True)
class BackgroundResearchJob:
    id: str
    query: str
    status: BackgroundResearchStatus
    created_at: datetime
    updated_at: datetime
    orchestrator_run_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", BackgroundResearchStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "updated_at", _aware_datetime("updated_at", self.updated_at))
        object.__setattr__(
            self,
            "orchestrator_run_id",
            _require_text("orchestrator_run_id", self.orchestrator_run_id),
        )
        object.__setattr__(self, "started_at", _optional_aware_datetime(self.started_at))
        object.__setattr__(self, "completed_at", _optional_aware_datetime(self.completed_at))
        object.__setattr__(self, "error_code", _optional_text(self.error_code))
        object.__setattr__(self, "error_message", _optional_text(self.error_message))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "orchestrator_run_id": self.orchestrator_run_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            id=str(payload["id"]),
            query=str(payload["query"]),
            status=BackgroundResearchStatus(str(payload["status"])),
            created_at=_datetime_from_payload(payload["created_at"]),
            updated_at=_datetime_from_payload(payload["updated_at"]),
            orchestrator_run_id=str(payload["orchestrator_run_id"]),
            started_at=_optional_datetime_from_payload(payload.get("started_at")),
            completed_at=_optional_datetime_from_payload(payload.get("completed_at")),
            error_code=_optional_payload_text(payload.get("error_code")),
            error_message=_optional_payload_text(payload.get("error_message")),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        )


ResearchCallable = Callable[[OrchestratorResearchInput], Awaitable[OrchestratedResearchRun]]


class BackgroundJobStore(Protocol):
    def save(self, job: BackgroundResearchJob) -> BackgroundResearchJob: ...

    def list(self) -> tuple[BackgroundResearchJob, ...]: ...

    def fail_unfinished(self, *, now: datetime) -> int: ...


class BackgroundResearchRunner:
    def __init__(
        self,
        *,
        max_concurrent_runs: int = 1,
        now: Callable[[], datetime] | None = None,
        job_store: BackgroundJobStore | None = None,
    ) -> None:
        if max_concurrent_runs <= 0:
            raise ValueError("max_concurrent_runs must be positive")
        self.max_concurrent_runs = max_concurrent_runs
        self._now = now or (lambda: datetime.now(UTC))
        self._job_store = job_store
        if self._job_store is not None:
            self._job_store.fail_unfinished(now=_aware_now(self._now()))
        stored_jobs = self._job_store.list() if self._job_store is not None else ()
        self._jobs: dict[str, BackgroundResearchJob] = {job.id: job for job in stored_jobs}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._lock = asyncio.Lock()

    async def submit(
        self,
        request: OrchestratorResearchInput,
        *,
        run: ResearchCallable,
        metadata: dict[str, str] | None = None,
        max_queued_runs: int | None = None,
    ) -> BackgroundResearchJob:
        if max_queued_runs is not None and max_queued_runs <= 0:
            raise ValueError("max_queued_runs must be positive")
        created_at = _aware_now(self._now())
        orchestrator_run_id = f"orchestrator_run_{uuid4().hex}"
        job = BackgroundResearchJob(
            id=f"background_research_{uuid4().hex}",
            query=request.query,
            status=BackgroundResearchStatus.QUEUED,
            created_at=created_at,
            updated_at=created_at,
            orchestrator_run_id=orchestrator_run_id,
            metadata=metadata or {},
        )
        request_with_run_id = replace(request, run_id=orchestrator_run_id)
        async with self._lock:
            if max_queued_runs is not None:
                queued_count = sum(
                    1
                    for existing in self._jobs.values()
                    if existing.status == BackgroundResearchStatus.QUEUED
                )
                if queued_count >= max_queued_runs:
                    raise BackgroundQueueFullError("background research queue is full")
            self._jobs[job.id] = job
            self._persist(job)
            self._tasks[job.id] = asyncio.create_task(
                self._execute(job.id, request_with_run_id, run),
                name=job.id,
            )
        return job

    async def cancel(self, job_id: str) -> BackgroundResearchJob | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in _TERMINAL_STATUSES:
                return job
            cancelled = self._replace_job(
                job,
                status=BackgroundResearchStatus.CANCELLED,
                completed_at=_aware_now(self._now()),
                error_code="cancelled",
                error_message="Background research run was cancelled.",
            )
            task = self._tasks.get(job_id)
            if task is not None:
                task.cancel()
            return cancelled

    async def get(self, job_id: str) -> BackgroundResearchJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(self) -> tuple[BackgroundResearchJob, ...]:
        async with self._lock:
            return tuple(sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True))

    async def stats(self) -> dict[str, object]:
        jobs = await self.list()
        return {
            "max_concurrent_runs": self.max_concurrent_runs,
            "job_count": len(jobs),
            "queued_count": sum(1 for job in jobs if job.status == BackgroundResearchStatus.QUEUED),
            "running_count": sum(
                1 for job in jobs if job.status == BackgroundResearchStatus.RUNNING
            ),
        }

    async def _execute(
        self,
        job_id: str,
        request: OrchestratorResearchInput,
        run: ResearchCallable,
    ) -> None:
        semaphore = self._get_semaphore()
        try:
            async with semaphore:
                async with self._lock:
                    job = self._jobs[job_id]
                    if job.status == BackgroundResearchStatus.CANCELLED:
                        return
                    self._replace_job(
                        job,
                        status=BackgroundResearchStatus.RUNNING,
                        started_at=_aware_now(self._now()),
                    )
                result = await run(request)
                async with self._lock:
                    job = self._jobs[job_id]
                    if job.status == BackgroundResearchStatus.CANCELLED:
                        return
                    self._replace_job(
                        job,
                        status=BackgroundResearchStatus.SUCCEEDED,
                        completed_at=_aware_now(self._now()),
                        warnings=result.warnings,
                    )
        except asyncio.CancelledError:
            async with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status != BackgroundResearchStatus.CANCELLED:
                    self._replace_job(
                        job,
                        status=BackgroundResearchStatus.CANCELLED,
                        completed_at=_aware_now(self._now()),
                        error_code="cancelled",
                        error_message="Background research run was cancelled.",
                    )
        except Exception as exc:
            async with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status != BackgroundResearchStatus.CANCELLED:
                    self._replace_job(
                        job,
                        status=BackgroundResearchStatus.FAILED,
                        completed_at=_aware_now(self._now()),
                        error_code=exc.__class__.__name__,
                        error_message=str(exc),
                    )
        finally:
            async with self._lock:
                if self._tasks.get(job_id) is asyncio.current_task():
                    self._tasks.pop(job_id, None)

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_runs)
        return self._semaphore

    def _replace_job(self, job: BackgroundResearchJob, **changes: object) -> BackgroundResearchJob:
        updated = replace(job, updated_at=_aware_now(self._now()), **changes)
        self._jobs[job.id] = updated
        self._persist(updated)
        return updated

    def _persist(self, job: BackgroundResearchJob) -> None:
        if self._job_store is not None:
            self._job_store.save(job)


_TERMINAL_STATUSES = {
    BackgroundResearchStatus.SUCCEEDED,
    BackgroundResearchStatus.FAILED,
    BackgroundResearchStatus.CANCELLED,
}


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _text_tuple(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a tuple of strings")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _optional_aware_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _aware_datetime("datetime", value)


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _aware_now(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _datetime_from_payload(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _optional_datetime_from_payload(value: object) -> datetime | None:
    return None if value is None else _datetime_from_payload(value)


def _optional_payload_text(value: object) -> str | None:
    return None if value is None else str(value)
