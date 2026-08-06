from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from financial_research_agent.observability import (
    RedactionPolicy,
    ResearchWorkItemStatus,
    build_research_work_view,
)
from financial_research_agent.orchestration import (
    AgentExecutionMetadata,
    AgentExecutionMode,
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)

NOW = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)


def test_work_projection_marks_the_parallel_refresh_stage_running() -> None:
    run = _run(handoffs=(_handoff("resolve_company", OrchestratorStepKind.COMPANY_RESOLUTION),))

    work = build_research_work_view(
        run,
        status="running",
        started_at=NOW,
        now=NOW + timedelta(seconds=12),
    )

    statuses = {item.kind: item.status for item in work.items}
    assert work.completed_steps == 1
    assert work.total_steps == 8
    assert work.elapsed_ms == 12_000
    assert statuses[OrchestratorStepKind.COMPANY_RESOLUTION] == ResearchWorkItemStatus.SUCCEEDED
    assert statuses[OrchestratorStepKind.MARKET_DATA_REFRESH] == ResearchWorkItemStatus.RUNNING
    assert (
        statuses[OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH] == ResearchWorkItemStatus.RUNNING
    )
    assert statuses[OrchestratorStepKind.FILING_REFRESH] == ResearchWorkItemStatus.RUNNING
    assert statuses[OrchestratorStepKind.SYNTHESIS] == ResearchWorkItemStatus.PENDING


def test_work_projection_exposes_only_safe_counts_and_activity() -> None:
    secret = "secret-value"
    handoff = _handoff(
        "context_analysis",
        OrchestratorStepKind.CONTEXT_ANALYSIS,
        evidence_ids=("evidence:1", "evidence:2"),
        warnings=("source warning",),
        limitations=("coverage limitation",),
        execution=AgentExecutionMetadata(
            mode=AgentExecutionMode.A2A,
            agent_role="context",
            correlation_id="private-correlation-id",
            provider="provider",
            model="model",
            reasoning_summary=f"hidden reasoning {secret}",
        ),
    )
    run = _run(
        plan=(default_orchestrator_plan()[6], default_orchestrator_plan()[7]),
        specialist_roles=("context", "synthesis"),
        handoffs=(handoff,),
    )

    payload = build_research_work_view(
        run,
        status="running",
        web_research_enabled=True,
        redaction_policy=RedactionPolicy(sensitive_values=(secret,)),
    ).to_dict()
    serialized = json.dumps(payload)

    context_item = payload["items"][0]
    assert context_item["evidence_count"] == 2
    assert context_item["warning_count"] == 1
    assert context_item["limitation_count"] == 1
    assert "Searching approved web sources" in context_item["activity"]
    assert "reasoning" not in serialized
    assert "private-correlation-id" not in serialized
    assert secret not in serialized
    assert "provider" not in serialized


def test_terminal_partial_work_keeps_step_statuses_and_skips_missing_steps() -> None:
    handoffs = (
        _handoff("resolve_company", OrchestratorStepKind.COMPANY_RESOLUTION),
        _handoff(
            "refresh_market_data",
            OrchestratorStepKind.MARKET_DATA_REFRESH,
            status=OrchestratorHandoffStatus.PARTIAL,
        ),
    )
    run = _run(
        status=OrchestratorRunStatus.PARTIAL,
        handoffs=handoffs,
        updated_at=NOW + timedelta(seconds=9),
    )

    work = build_research_work_view(run, status="succeeded")

    assert work.status == "succeeded"
    assert work.elapsed_ms == 9_000
    assert work.items[1].status == ResearchWorkItemStatus.PARTIAL
    assert work.items[-1].status == ResearchWorkItemStatus.SKIPPED
    assert work.completed_steps == work.total_steps


def test_failed_work_marks_first_unfinished_step_failed() -> None:
    run = _run(handoffs=(_handoff("resolve_company", OrchestratorStepKind.COMPANY_RESOLUTION),))

    work = build_research_work_view(run, status="failed")

    assert work.items[1].status == ResearchWorkItemStatus.FAILED
    assert all(item.status == ResearchWorkItemStatus.SKIPPED for item in work.items[2:])


def _run(
    *,
    status: OrchestratorRunStatus = OrchestratorRunStatus.RUNNING,
    plan=None,
    specialist_roles=("financial-report", "stock", "context", "synthesis"),
    handoffs: tuple[AgentHandoff, ...] = (),
    updated_at: datetime = NOW,
) -> OrchestratedResearchRun:
    return OrchestratedResearchRun(
        id="orchestrator_run_test",
        query="TEST research query",
        status=status,
        created_at=NOW,
        updated_at=updated_at,
        execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
        plan=tuple(plan or default_orchestrator_plan(specialist_roles)),
        specialist_roles=specialist_roles,
        handoffs=handoffs,
    )


def _handoff(
    step_id: str,
    kind: OrchestratorStepKind,
    *,
    status: OrchestratorHandoffStatus = OrchestratorHandoffStatus.SUCCEEDED,
    evidence_ids: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    execution: AgentExecutionMetadata | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff_{step_id}",
        step_id=step_id,
        kind=kind,
        status=status,
        started_at=NOW,
        completed_at=NOW + timedelta(milliseconds=125),
        evidence_ids=evidence_ids,
        warnings=warnings,
        limitations=limitations,
        confidence=HandoffConfidence.HIGH,
        execution=execution,
    )
