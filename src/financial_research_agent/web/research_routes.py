from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from financial_research_agent.background import (
    BackgroundResearchJob,
    BackgroundResearchRunner,
    BackgroundResearchStatus,
)
from financial_research_agent.observability import (
    RedactionPolicy,
    build_research_work_view,
    build_trace_from_orchestrator_run,
    status_from_run,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorRunStatus,
    OrchestratorRunStore,
    default_orchestrator_plan,
)
from financial_research_agent.report_exports import build_report_evidence_index
from financial_research_agent.settings import Settings


def create_research_router(
    *,
    background_research: BackgroundResearchRunner,
    orchestrator_runs: OrchestratorRunStore,
    settings: Callable[[], Settings],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/background/research-runs")
    async def list_background_research_runs() -> dict[str, Any]:
        return {
            "jobs": [
                background_job_payload(job, orchestrator_runs, settings=settings())["job"]
                for job in await background_research.list()
            ],
            "limits": await background_research.stats(),
        }

    @router.get("/api/background/research-runs/{job_id}")
    async def get_background_research_run(job_id: str) -> dict[str, Any]:
        job = await background_research.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "background_job_not_found"})
        return background_job_payload(job, orchestrator_runs, settings=settings())

    @router.post("/api/background/research-runs/{job_id}/cancel")
    async def cancel_background_research_run(job_id: str) -> dict[str, Any]:
        job = await background_research.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "background_job_not_found"})
        mark_cancelled_orchestrator_run(job, orchestrator_runs)
        return background_job_payload(job, orchestrator_runs, settings=settings())

    @router.get("/api/orchestrator/runs")
    def list_orchestrator_runs() -> dict[str, Any]:
        return {
            "runs": [
                {
                    **run.to_dict(),
                    "synthesis_report": synthesis_report_from_run(run),
                    "cited_context_answer": (
                        dict(run.cited_context_answer)
                        if run.cited_context_answer is not None
                        else None
                    ),
                }
                for run in orchestrator_runs.list()
            ]
        }

    @router.get("/api/orchestrator/runs/{run_id}")
    async def get_orchestrator_run(run_id: str) -> dict[str, Any]:
        run = orchestrator_run_or_404(orchestrator_runs, run_id)
        background_job = next(
            (job for job in await background_research.list() if job.orchestrator_run_id == run_id),
            None,
        )
        return {
            "run": run.to_dict(),
            "work": (
                _job_work_payload(background_job, run, settings())
                if background_job is not None
                else _run_work_payload(run, settings())
            ),
            "synthesis_report": synthesis_report_from_run(run),
            "cited_context_answer": (
                dict(run.cited_context_answer) if run.cited_context_answer is not None else None
            ),
        }

    @router.get("/api/orchestrator/runs/{run_id}/evidence")
    def get_orchestrator_run_evidence(run_id: str) -> dict[str, Any]:
        run = orchestrator_run_or_404(orchestrator_runs, run_id)
        evidence = build_report_evidence_index(
            run,
            redaction_policy=RedactionPolicy.from_settings(settings()),
        )
        return {"evidence": evidence.to_dict()}

    @router.get("/api/orchestrator/runs/{run_id}/trace")
    def get_orchestrator_run_trace(run_id: str) -> dict[str, Any]:
        run = orchestrator_run_or_404(orchestrator_runs, run_id)
        trace = build_trace_from_orchestrator_run(
            run,
            redaction_policy=RedactionPolicy.from_settings(settings()),
        )
        return {"trace": trace.to_dict()}

    return router


def synthesis_report_from_run(run: OrchestratedResearchRun) -> dict[str, object] | None:
    if run.research_subject.value == "general_context":
        return None
    for handoff in reversed(run.handoffs):
        if handoff.kind.value != "synthesis":
            continue
        report = handoff.output.get("report")
        return report if isinstance(report, dict) else None
    return None


def orchestrator_run_or_404(
    orchestrator_runs: OrchestratorRunStore,
    run_id: str,
) -> OrchestratedResearchRun:
    run = orchestrator_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "orchestrator_run_not_found"})
    return run


def background_job_payload(
    job: BackgroundResearchJob,
    orchestrator_runs: OrchestratorRunStore,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    run = orchestrator_runs.get(job.orchestrator_run_id)
    work = _job_work_payload(job, run, settings)
    progress = _orchestrator_progress(run) if run is not None else _empty_progress()
    progress["work"] = work
    return {
        "job": {
            **job.to_dict(),
            "progress": progress,
            "orchestrator_run": run.to_dict() if run is not None else None,
            "synthesis_report": synthesis_report_from_run(run) if run is not None else None,
            "cited_context_answer": (
                dict(run.cited_context_answer)
                if run is not None and run.cited_context_answer is not None
                else None
            ),
        }
    }


def mark_cancelled_orchestrator_run(
    job: BackgroundResearchJob,
    orchestrator_runs: OrchestratorRunStore,
) -> None:
    if job.status != BackgroundResearchStatus.CANCELLED:
        return
    run = orchestrator_runs.get(job.orchestrator_run_id)
    if run is None or run.status != OrchestratorRunStatus.RUNNING:
        return
    limitation = "Background research job was cancelled before the workflow completed."
    orchestrator_runs.save(
        replace(
            run,
            status=OrchestratorRunStatus.PARTIAL,
            limitations=tuple(dict.fromkeys((*run.limitations, limitation))),
            updated_at=datetime.now(UTC),
        )
    )


def _empty_progress() -> dict[str, object]:
    return {
        "completed_steps": 0,
        "total_steps": len(default_orchestrator_plan()),
        "current_step": None,
    }


def _orchestrator_progress(run: OrchestratedResearchRun) -> dict[str, object]:
    completed_steps = {handoff.step_id for handoff in run.handoffs}
    remaining_steps = tuple(step.id for step in run.plan if step.id not in completed_steps)
    return {
        "completed_steps": len(completed_steps),
        "total_steps": len(run.plan),
        "current_step": remaining_steps[0] if remaining_steps else None,
    }


def _run_work_payload(run: OrchestratedResearchRun, settings: Settings) -> dict[str, object]:
    return build_research_work_view(
        run,
        status=status_from_run(run),
        web_research_enabled=settings.data_sources.web_research_enabled,
        redaction_policy=RedactionPolicy.from_settings(settings),
    ).to_dict()


def _job_work_payload(
    job: BackgroundResearchJob,
    run: OrchestratedResearchRun | None,
    settings: Settings | None,
) -> dict[str, object]:
    redaction_policy = RedactionPolicy.from_settings(settings) if settings else RedactionPolicy()
    return build_research_work_view(
        run,
        status=job.status.value,
        started_at=job.started_at,
        completed_at=job.completed_at,
        web_research_enabled=bool(settings and settings.data_sources.web_research_enabled),
        redaction_policy=redaction_policy,
    ).to_dict()
