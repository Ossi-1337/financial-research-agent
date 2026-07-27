from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime

from financial_research_agent.a2a import (
    A2AResearchRuntime,
    A2AResearchStepDispatcher,
    SQLiteA2ATaskStore,
    create_a2a_app,
)
from financial_research_agent.background import BackgroundResearchRunner
from financial_research_agent.orchestration import (
    AgentEndpoint,
    AgentHandoff,
    AgentRole,
    DelegationRequest,
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


class FixtureSpecialistService:
    async def execute(self, request: DelegationRequest) -> AgentHandoff:
        now = datetime.now(UTC)
        return AgentHandoff(
            id=f"handoff:{request.run_id}:{request.step_id}",
            step_id=request.step_id,
            kind=request.expected_kind,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=now,
            completed_at=now,
            output={
                "summary": "TEST FIXTURE SYNTHESIS"
                if request.role == AgentRole.SYNTHESIS
                else None,
                "analysis": {"fixture": "TEST TOOL OUTPUT"},
            },
            evidence_ids=(f"fixture:{request.role.value}:1",),
            confidence=HandoffConfidence.HIGH,
        )


class FixtureDistributedOrchestrator:
    def __init__(self, run_store: object, dispatcher: A2AResearchStepDispatcher) -> None:
        self.run_store = run_store
        self.dispatcher = dispatcher

    async def run(self, request, *, progress_observer=None) -> OrchestratedResearchRun:
        now = datetime.now(UTC)
        run = OrchestratedResearchRun(
            id=request.run_id,
            query=request.query,
            status=OrchestratorRunStatus.RUNNING,
            created_at=now,
            updated_at=now,
            execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
            plan=default_orchestrator_plan(),
        )
        self.run_store.save(run)
        specialist_requests = (
            _delegation(run.id, AgentRole.FINANCIAL_REPORT),
            _delegation(run.id, AgentRole.STOCK),
            _delegation(run.id, AgentRole.CONTEXT),
        )
        results = await asyncio.gather(
            *(self.dispatcher.dispatch(item, run=run) for item in specialist_requests)
        )
        run = replace(
            run,
            handoffs=tuple(result.handoff for result in results),
            updated_at=datetime.now(UTC),
        )
        self.run_store.save(run)
        synthesis = await self.dispatcher.dispatch(
            _delegation(run.id, AgentRole.SYNTHESIS),
            run=run,
        )
        all_handoffs = (*run.handoffs, synthesis.handoff)
        complete = all(
            handoff.status == OrchestratorHandoffStatus.SUCCEEDED for handoff in all_handoffs
        )
        run = replace(
            run,
            status=(OrchestratorRunStatus.COMPLETE if complete else OrchestratorRunStatus.PARTIAL),
            handoffs=all_handoffs,
            synthesis_summary="TEST FIXTURE SYNTHESIS",
            warnings=("TEST FIXTURE OUTPUT ONLY.",),
            limitations=("No live providers used.",),
            updated_at=datetime.now(UTC),
        )
        self.run_store.save(run)
        if progress_observer is not None:
            progress_observer(run)
        return run


def _delegation(run_id: str, role: AgentRole) -> DelegationRequest:
    definitions = {
        AgentRole.FINANCIAL_REPORT: (
            "financial_report_analysis",
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            {"company_id": "fixture-company", "legal_name": "Fixture Inc.", "cik": "1"},
        ),
        AgentRole.STOCK: (
            "stock_price_analysis",
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            {"security_id": "fixture-security", "ticker": "TEST"},
        ),
        AgentRole.CONTEXT: (
            "context_analysis",
            OrchestratorStepKind.CONTEXT_ANALYSIS,
            {"query": "TEST FIXTURE query", "company_symbols": ["TEST"], "source_items": []},
        ),
        AgentRole.SYNTHESIS: (
            "synthesis",
            OrchestratorStepKind.SYNTHESIS,
            {"handoff_ids": []},
        ),
    }
    step_id, kind, payload = definitions[role]
    return DelegationRequest(
        role=role,
        run_id=run_id,
        step_id=step_id,
        correlation_id=run_id,
        expected_kind=kind,
        payload=payload,
    )


settings = Settings.from_env()
role = AgentRole(os.environ["FRA_A2A_ROLE"])
persistence = create_persistence(settings)
assert persistence.database is not None
specialist_service = FixtureSpecialistService()
if role == AgentRole.COMPANY_RESEARCH:
    dispatcher = A2AResearchStepDispatcher(
        endpoints={
            AgentRole.FINANCIAL_REPORT: AgentEndpoint(
                AgentRole.FINANCIAL_REPORT,
                "financial-report",
                settings.a2a.financial_report_url,
                "financial_report_analysis",
            ),
            AgentRole.STOCK: AgentEndpoint(
                AgentRole.STOCK,
                "stock",
                settings.a2a.stock_url,
                "stock_price_analysis",
            ),
            AgentRole.CONTEXT: AgentEndpoint(
                AgentRole.CONTEXT,
                "context",
                settings.a2a.context_url,
                "context_analysis",
            ),
            AgentRole.SYNTHESIS: AgentEndpoint(
                AgentRole.SYNTHESIS,
                "synthesis",
                settings.a2a.synthesis_url,
                "research_synthesis",
            ),
        },
        timeout_seconds=10,
        max_attempts=2,
    )
    orchestrator = FixtureDistributedOrchestrator(
        persistence.orchestrator_runs,
        dispatcher,
    )
else:
    orchestrator = object()
runtime = A2AResearchRuntime(
    orchestrator=orchestrator,  # type: ignore[arg-type]
    background_runner=BackgroundResearchRunner(
        max_concurrent_runs=1,
        job_store=persistence.background_jobs,
    ),
    task_store=SQLiteA2ATaskStore(persistence.database, owner=role.value),
    orchestrator_run_store=persistence.orchestrator_runs,
    persistence=persistence,
    role=role,
    specialist_service=specialist_service,  # type: ignore[arg-type]
)
app = create_a2a_app(settings=settings, runtime=runtime, role=role)
