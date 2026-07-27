from __future__ import annotations

from datetime import UTC, datetime

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


class FixtureOrchestrator:
    def __init__(self, run_store: object) -> None:
        self.run_store = run_store

    async def run(self, request, *, progress_observer=None) -> OrchestratedResearchRun:
        now = datetime.now(UTC)
        handoff = AgentHandoff(
            id=f"handoff:{request.run_id}:synthesis",
            step_id="synthesis",
            kind=OrchestratorStepKind.SYNTHESIS,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=now,
            completed_at=now,
            output={"report": {"summary": "TEST FIXTURE OUTPUT"}},
            evidence_ids=("fixture:evidence:1",),
            confidence=HandoffConfidence.MEDIUM,
        )
        run = OrchestratedResearchRun(
            id=request.run_id,
            query=request.query,
            status=OrchestratorRunStatus.COMPLETE,
            created_at=now,
            updated_at=now,
            execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
            plan=default_orchestrator_plan(),
            handoffs=(handoff,),
            synthesis_summary="TEST FIXTURE OUTPUT",
            warnings=("TEST FIXTURE OUTPUT ONLY.",),
            limitations=("No live providers used.",),
        )
        self.run_store.save(run)
        if progress_observer is not None:
            progress_observer(run)
        return run


settings = Settings.from_env()
persistence = create_persistence(settings)
assert persistence.database is not None
orchestrator = FixtureOrchestrator(persistence.orchestrator_runs)
runtime = A2AResearchRuntime(
    orchestrator=orchestrator,
    background_runner=BackgroundResearchRunner(
        max_concurrent_runs=1,
        job_store=persistence.background_jobs,
    ),
    task_store=SQLiteA2ATaskStore(persistence.database),
    orchestrator_run_store=persistence.orchestrator_runs,
    persistence=persistence,
)
app = create_a2a_app(settings=settings, runtime=runtime)
