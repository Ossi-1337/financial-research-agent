from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from financial_research_agent.observability.contracts import RedactionPolicy
from financial_research_agent.orchestration import (
    AgentHandoff,
    OrchestratedResearchRun,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)

WORK_VIEW_SCHEMA_VERSION = 1


class ResearchWorkItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ResearchWorkItem:
    kind: OrchestratorStepKind
    title: str
    activity: str
    status: ResearchWorkItemStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    evidence_count: int = 0
    warning_count: int = 0
    limitation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OrchestratorStepKind(self.kind))
        object.__setattr__(self, "status", ResearchWorkItemStatus(self.status))
        object.__setattr__(self, "title", _required_text("title", self.title))
        object.__setattr__(self, "activity", _required_text("activity", self.activity))
        for name in ("evidence_count", "warning_count", "limitation_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "activity": self.activity,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "evidence_count": self.evidence_count,
            "warning_count": self.warning_count,
            "limitation_count": self.limitation_count,
        }


@dataclass(frozen=True, slots=True)
class ResearchWorkView:
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    elapsed_ms: int
    completed_steps: int
    total_steps: int
    current_step: str | None
    items: tuple[ResearchWorkItem, ...]
    schema_version: int = WORK_VIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _required_text("status", self.status))
        object.__setattr__(self, "items", tuple(self.items))
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        if self.completed_steps < 0 or self.total_steps < 0:
            raise ValueError("step counts must be non-negative")
        if self.completed_steps > self.total_steps:
            raise ValueError("completed_steps cannot exceed total_steps")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_ms": self.elapsed_ms,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "items": [item.to_dict() for item in self.items],
        }


def build_research_work_view(
    run: OrchestratedResearchRun | None,
    *,
    status: str,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    now: datetime | None = None,
    web_research_enabled: bool = False,
    redaction_policy: RedactionPolicy | None = None,
) -> ResearchWorkView:
    current_time = _aware(now or datetime.now(UTC))
    policy = redaction_policy or RedactionPolicy()
    plan = run.plan if run is not None else default_orchestrator_plan()
    handoffs = {handoff.step_id: handoff for handoff in run.handoffs} if run else {}
    active_kinds = _active_kinds(run, status)
    missing_failure_kind = _missing_failure_kind(run, status)
    items = tuple(
        _work_item(
            step.kind,
            str(policy.redact(step.title)),
            handoffs.get(step.id),
            active=step.kind in active_kinds,
            failed=step.kind == missing_failure_kind,
            terminal=status in {"succeeded", "failed", "cancelled"},
            web_research_enabled=web_research_enabled,
        )
        for step in plan
    )
    effective_started_at = started_at or (run.created_at if run is not None else None)
    effective_completed_at = completed_at
    if (
        effective_completed_at is None
        and run is not None
        and run.status != OrchestratorRunStatus.RUNNING
    ):
        effective_completed_at = run.updated_at
    elapsed_end = effective_completed_at or current_time
    elapsed_ms = _duration_ms(effective_started_at, elapsed_end) or 0
    completed = sum(
        item.status
        in {
            ResearchWorkItemStatus.SUCCEEDED,
            ResearchWorkItemStatus.PARTIAL,
            ResearchWorkItemStatus.SKIPPED,
            ResearchWorkItemStatus.FAILED,
        }
        for item in items
    )
    current_item = next(
        (item for item in items if item.status == ResearchWorkItemStatus.RUNNING),
        None,
    )
    return ResearchWorkView(
        status=status,
        started_at=effective_started_at,
        completed_at=effective_completed_at,
        elapsed_ms=elapsed_ms,
        completed_steps=completed,
        total_steps=len(items),
        current_step=current_item.kind.value if current_item else None,
        items=items,
    )


def status_from_run(run: OrchestratedResearchRun) -> str:
    if run.status == OrchestratorRunStatus.RUNNING:
        return "running"
    if run.status == OrchestratorRunStatus.FAILED:
        return "failed"
    return "succeeded"


def _work_item(
    kind: OrchestratorStepKind,
    title: str,
    handoff: AgentHandoff | None,
    *,
    active: bool,
    failed: bool,
    terminal: bool,
    web_research_enabled: bool,
) -> ResearchWorkItem:
    if handoff is not None:
        status = ResearchWorkItemStatus(handoff.status.value)
        return ResearchWorkItem(
            kind=kind,
            title=title,
            activity=_activity(kind, web_research_enabled),
            status=status,
            started_at=handoff.started_at,
            completed_at=handoff.completed_at,
            duration_ms=_duration_ms(handoff.started_at, handoff.completed_at),
            evidence_count=len(handoff.evidence_ids),
            warning_count=len(handoff.warnings),
            limitation_count=len(handoff.limitations),
        )
    if failed:
        item_status = ResearchWorkItemStatus.FAILED
    elif active:
        item_status = ResearchWorkItemStatus.RUNNING
    elif terminal:
        item_status = ResearchWorkItemStatus.SKIPPED
    else:
        item_status = ResearchWorkItemStatus.PENDING
    return ResearchWorkItem(
        kind=kind,
        title=title,
        activity=_activity(kind, web_research_enabled),
        status=item_status,
    )


def _active_kinds(run: OrchestratedResearchRun | None, status: str) -> set[OrchestratorStepKind]:
    if run is None or status != "running":
        return set()
    completed = {handoff.kind for handoff in run.handoffs}
    stages = (
        {OrchestratorStepKind.COMPANY_RESOLUTION},
        {
            OrchestratorStepKind.MARKET_DATA_REFRESH,
            OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
            OrchestratorStepKind.FILING_REFRESH,
        },
        {
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            OrchestratorStepKind.CONTEXT_ANALYSIS,
        },
        {OrchestratorStepKind.SYNTHESIS},
    )
    planned = {step.kind for step in run.plan}
    for stage in stages:
        pending = (stage & planned) - completed
        if pending:
            return pending
    return set()


def _missing_failure_kind(
    run: OrchestratedResearchRun | None,
    status: str,
) -> OrchestratorStepKind | None:
    if status != "failed":
        return None
    if run is None:
        return default_orchestrator_plan()[0].kind
    completed = {handoff.kind for handoff in run.handoffs}
    return next((step.kind for step in run.plan if step.kind not in completed), None)


def _activity(kind: OrchestratorStepKind, web_research_enabled: bool) -> str:
    if kind == OrchestratorStepKind.COMPANY_RESOLUTION:
        return "Resolving the company and primary security."
    if kind == OrchestratorStepKind.MARKET_DATA_REFRESH:
        return "Fetching company and benchmark market data."
    if kind == OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH:
        return "Fetching financial statements."
    if kind == OrchestratorStepKind.FILING_REFRESH:
        return "Fetching regulatory filings."
    if kind == OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS:
        return "Searching stored statement and filing evidence, then reviewing financials."
    if kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS:
        return "Calculating company and benchmark metrics, then reviewing stock behavior."
    if kind == OrchestratorStepKind.CONTEXT_ANALYSIS:
        if web_research_enabled:
            return "Searching approved web sources and reviewing company context."
        return "Reviewing approved stored context sources."
    return "Validating specialist handoffs and composing the source-backed report."


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, round((_aware(end) - _aware(start)).total_seconds() * 1000))


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def _required_text(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text
