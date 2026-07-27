from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.background import (
    BackgroundQueueFullError,
    BackgroundResearchJob,
    BackgroundResearchRunner,
    BackgroundResearchStatus,
)
from financial_research_agent.entities import (
    CompanySearchResult,
    CompanySearchStatus,
    SourceMetadata,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    default_orchestrator_plan,
)

NOW = datetime(2026, 7, 7, 12, tzinfo=UTC)


def test_background_job_contract_is_immutable_and_serializes() -> None:
    job = BackgroundResearchJob(
        id="background_research_test",
        query="Apple research",
        status=BackgroundResearchStatus.QUEUED,
        created_at=NOW,
        updated_at=NOW,
        orchestrator_run_id="orchestrator_run_test",
    )

    assert job.to_dict()["status"] == "queued"
    with pytest.raises(FrozenInstanceError):
        job.status = BackgroundResearchStatus.RUNNING  # type: ignore[misc]
    with pytest.raises(ValueError, match="query is required"):
        BackgroundResearchJob(
            id="background_research_test",
            query=" ",
            status=BackgroundResearchStatus.QUEUED,
            created_at=NOW,
            updated_at=NOW,
            orchestrator_run_id="orchestrator_run_test",
        )


def test_background_runner_limits_concurrent_research_runs() -> None:
    async def scenario() -> None:
        started: list[str] = []
        release = asyncio.Event()
        runner = BackgroundResearchRunner(max_concurrent_runs=1, now=lambda: NOW)

        async def run(request: OrchestratorResearchInput) -> OrchestratedResearchRun:
            started.append(request.query)
            await release.wait()
            return _run(request)

        first = await runner.submit(OrchestratorResearchInput(query="first"), run=run)
        second = await runner.submit(OrchestratorResearchInput(query="second"), run=run)
        await _wait_for(lambda: len(started) == 1)
        stats = await runner.stats()

        assert first.status == BackgroundResearchStatus.QUEUED
        assert second.status == BackgroundResearchStatus.QUEUED
        assert stats["running_count"] == 1
        assert stats["queued_count"] == 1

        release.set()
        await _wait_for_job_statuses(runner, BackgroundResearchStatus.SUCCEEDED)

    asyncio.run(scenario())


def test_background_runner_cancels_queued_or_running_job() -> None:
    async def scenario() -> None:
        runner = BackgroundResearchRunner(max_concurrent_runs=1, now=lambda: NOW)

        async def run(request: OrchestratorResearchInput) -> OrchestratedResearchRun:
            await asyncio.sleep(30)
            return _run(request)

        job = await runner.submit(OrchestratorResearchInput(query="cancel me"), run=run)
        cancelled = await runner.cancel(job.id)

        assert cancelled is not None
        assert cancelled.status == BackgroundResearchStatus.CANCELLED
        assert (await runner.get(job.id)).status == BackgroundResearchStatus.CANCELLED  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_background_runner_admits_queued_jobs_atomically() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        runner = BackgroundResearchRunner(max_concurrent_runs=1, now=lambda: NOW)

        async def run(request: OrchestratorResearchInput) -> OrchestratedResearchRun:
            await release.wait()
            return _run(request)

        running = await runner.submit(OrchestratorResearchInput(query="running"), run=run)
        await _wait_for_running_job(runner)
        results = await asyncio.gather(
            *(
                runner.submit(
                    OrchestratorResearchInput(query=f"queued-{index}"),
                    run=run,
                    max_queued_runs=1,
                )
                for index in range(20)
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(result, BackgroundResearchJob) for result in results) == 1
        assert sum(isinstance(result, BackgroundQueueFullError) for result in results) == 19
        stats = await runner.stats()
        assert stats["running_count"] == 1
        assert stats["queued_count"] == 1
        await runner.cancel(running.id)
        for job in await runner.list():
            await runner.cancel(job.id)

    asyncio.run(scenario())


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


async def _wait_for_job_statuses(
    runner: BackgroundResearchRunner,
    status: BackgroundResearchStatus,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = await runner.list()
        if jobs and all(job.status == status for job in jobs):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("jobs did not reach expected status before timeout")


async def _wait_for_running_job(
    runner: BackgroundResearchRunner,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (await runner.stats())["running_count"] == 1:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("job did not start before timeout")


def _poll_job(client: TestClient, job_id: str, *, timeout: float = 2.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/background/research-runs/{job_id}").json()["job"]
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError("background job did not finish before timeout")


def _run(request: OrchestratorResearchInput) -> OrchestratedResearchRun:
    return OrchestratedResearchRun(
        id=request.run_id or "orchestrator_run_test",
        query=request.query,
        status=OrchestratorRunStatus.COMPLETE,
        created_at=NOW,
        updated_at=NOW,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=default_orchestrator_plan(),
    )


class NoMatchCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.NO_MATCHES,
            candidates=(),
            source=SourceMetadata(
                provider="fake-company-search",
                provider_status=f"test fixture limit={limit}",
                source_url="https://example.test/company-search",
                retrieved_at=NOW,
                attribution="test fixture",
            ),
        )


class SlowNoMatchCompanySearchProvider(NoMatchCompanySearchProvider):
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        await asyncio.sleep(30)
        return await super().search(query, limit=limit)
