from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from financial_research_agent.context_analysis import ContextSourceItem

NO_ORCHESTRATOR_RECOMMENDATION_NOTICE = (
    "This orchestrated research workflow coordinates source-backed research steps only and "
    "does not provide buy, sell, hold, price-target, or personalized investment advice."
)

ALLOWED_RESEARCH_SPECIALIST_ROLES = (
    "financial-report",
    "stock",
    "context",
    "synthesis",
)
DEFAULT_RESEARCH_SPECIALIST_ROLES = ALLOWED_RESEARCH_SPECIALIST_ROLES


class OrchestratorExecutionPolicy(StrEnum):
    SEQUENTIAL_LOCAL_SAFE = "sequential_local_safe"
    DISTRIBUTED_A2A = "distributed_a2a"


class OrchestratorRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class OrchestratorHandoffStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class OrchestratorStepKind(StrEnum):
    COMPANY_RESOLUTION = "company_resolution"
    MARKET_DATA_REFRESH = "market_data_refresh"
    FINANCIAL_STATEMENT_REFRESH = "financial_statement_refresh"
    FILING_REFRESH = "filing_refresh"
    FINANCIAL_REPORT_ANALYSIS = "financial_report_analysis"
    STOCK_PRICE_ANALYSIS = "stock_price_analysis"
    CONTEXT_ANALYSIS = "context_analysis"
    SYNTHESIS = "synthesis"


class HandoffConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class AgentExecutionMode(StrEnum):
    LOCAL = "local"
    A2A = "a2a"


@dataclass(frozen=True, slots=True)
class AgentExecutionMetadata:
    mode: AgentExecutionMode
    agent_role: str
    correlation_id: str
    delegation_id: str | None = None
    remote_task_id: str | None = None
    service_id: str | None = None
    attempt_count: int = 1
    prompt_id: str | None = None
    prompt_version: str | None = None
    provider: str | None = None
    model: str | None = None
    tool_status: str | None = None
    reasoning_summary: str | None = None
    skill_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AgentExecutionMode(self.mode))
        object.__setattr__(self, "agent_role", _require_text("agent_role", self.agent_role))
        object.__setattr__(
            self,
            "correlation_id",
            _require_text("correlation_id", self.correlation_id),
        )
        object.__setattr__(self, "delegation_id", _optional_text(self.delegation_id))
        object.__setattr__(self, "remote_task_id", _optional_text(self.remote_task_id))
        object.__setattr__(self, "service_id", _optional_text(self.service_id))
        object.__setattr__(self, "prompt_id", _optional_text(self.prompt_id))
        object.__setattr__(self, "prompt_version", _optional_text(self.prompt_version))
        object.__setattr__(self, "provider", _optional_text(self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))
        object.__setattr__(self, "tool_status", _optional_text(self.tool_status))
        object.__setattr__(self, "reasoning_summary", _optional_text(self.reasoning_summary))
        object.__setattr__(
            self,
            "skill_references",
            _text_tuple("skill_references", self.skill_references),
        )
        if self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "agent_role": self.agent_role,
            "correlation_id": self.correlation_id,
            "delegation_id": self.delegation_id,
            "remote_task_id": self.remote_task_id,
            "service_id": self.service_id,
            "attempt_count": self.attempt_count,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "tool_status": self.tool_status,
            "reasoning_summary": self.reasoning_summary,
            "skill_references": list(self.skill_references),
        }


@dataclass(frozen=True, slots=True)
class OrchestratorResearchInput:
    query: str
    company_query: str | None = None
    run_id: str | None = None
    refresh: bool = True
    company_search_limit: int = 3
    fiscal_years: int = 3
    filing_forms: tuple[str, ...] = ("10-K", "10-Q", "20-F", "6-K")
    filing_limit: int = 1
    filing_form_limits: Mapping[str, int] = field(default_factory=dict)
    market_outputsize: str = "compact"
    benchmark_symbol: str | None = None
    context_source_items: tuple[ContextSourceItem, ...] = ()
    specialist_roles: tuple[str, ...] = DEFAULT_RESEARCH_SPECIALIST_ROLES
    agent_provider: str | None = None
    agent_model: str | None = None
    orchestrator_skill_references: tuple[str, ...] = ()
    scenario_id: str | None = None
    scenario_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "company_query", _optional_text(self.company_query))
        object.__setattr__(self, "run_id", _optional_text(self.run_id))
        if self.company_search_limit <= 0:
            raise ValueError("company_search_limit must be positive")
        if self.fiscal_years <= 0:
            raise ValueError("fiscal_years must be positive")
        if self.filing_limit <= 0:
            raise ValueError("filing_limit must be positive")
        object.__setattr__(self, "filing_forms", _text_tuple("filing_forms", self.filing_forms))
        object.__setattr__(
            self,
            "filing_form_limits",
            _positive_int_mapping("filing_form_limits", self.filing_form_limits),
        )
        object.__setattr__(
            self,
            "market_outputsize",
            _require_text("market_outputsize", self.market_outputsize),
        )
        object.__setattr__(self, "benchmark_symbol", _optional_upper_text(self.benchmark_symbol))
        object.__setattr__(self, "scenario_id", _optional_text(self.scenario_id))
        object.__setattr__(self, "scenario_version", _optional_text(self.scenario_version))
        if (self.scenario_id is None) != (self.scenario_version is None):
            raise ValueError("scenario_id and scenario_version must be provided together")
        object.__setattr__(
            self,
            "context_source_items",
            _context_source_item_tuple(self.context_source_items),
        )
        object.__setattr__(
            self,
            "specialist_roles",
            _specialist_role_tuple(self.specialist_roles),
        )
        object.__setattr__(self, "agent_provider", _optional_text(self.agent_provider))
        object.__setattr__(self, "agent_model", _optional_text(self.agent_model))
        object.__setattr__(
            self,
            "orchestrator_skill_references",
            _text_tuple(
                "orchestrator_skill_references",
                self.orchestrator_skill_references,
            ),
        )
        if (self.agent_provider is None) != (self.agent_model is None):
            raise ValueError("agent_provider and agent_model must be provided together")


@dataclass(frozen=True, slots=True)
class OrchestratorPlanStep:
    id: str
    kind: OrchestratorStepKind
    title: str
    required: bool
    can_run_parallel: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "kind", OrchestratorStepKind(self.kind))
        object.__setattr__(self, "title", _require_text("title", self.title))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "required": self.required,
            "can_run_parallel": self.can_run_parallel,
        }


@dataclass(frozen=True, slots=True)
class AgentHandoff:
    id: str
    step_id: str
    kind: OrchestratorStepKind
    status: OrchestratorHandoffStatus
    started_at: datetime
    completed_at: datetime
    input_summary: Mapping[str, str] = field(default_factory=dict)
    output: Mapping[str, object] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    confidence: HandoffConfidence = HandoffConfidence.UNKNOWN
    error_code: str | None = None
    error_message: str | None = None
    execution: AgentExecutionMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "step_id", _require_text("step_id", self.step_id))
        object.__setattr__(self, "kind", OrchestratorStepKind(self.kind))
        object.__setattr__(self, "status", OrchestratorHandoffStatus(self.status))
        object.__setattr__(self, "started_at", _aware_datetime("started_at", self.started_at))
        object.__setattr__(self, "completed_at", _aware_datetime("completed_at", self.completed_at))
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be greater than or equal to started_at")
        object.__setattr__(
            self,
            "input_summary",
            _text_mapping("input_summary", self.input_summary),
        )
        object.__setattr__(self, "output", _object_mapping("output", self.output))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(self, "confidence", HandoffConfidence(self.confidence))
        object.__setattr__(self, "error_code", _optional_text(self.error_code))
        object.__setattr__(self, "error_message", _optional_text(self.error_message))
        if self.execution is not None and not isinstance(
            self.execution,
            AgentExecutionMetadata,
        ):
            raise ValueError("execution must be AgentExecutionMetadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "input_summary": dict(self.input_summary),
            "output": dict(self.output),
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "confidence": self.confidence.value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "execution": self.execution.to_dict() if self.execution is not None else None,
        }


@dataclass(frozen=True, slots=True)
class OrchestratedResearchRun:
    id: str
    query: str
    status: OrchestratorRunStatus
    created_at: datetime
    updated_at: datetime
    execution_policy: OrchestratorExecutionPolicy
    plan: tuple[OrchestratorPlanStep, ...]
    specialist_roles: tuple[str, ...] = DEFAULT_RESEARCH_SPECIALIST_ROLES
    agent_provider: str | None = None
    agent_model: str | None = None
    orchestrator_skill_references: tuple[str, ...] = ()
    handoffs: tuple[AgentHandoff, ...] = ()
    selected_company: Mapping[str, object] | None = None
    selected_security: Mapping[str, object] | None = None
    synthesis_summary: str | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    no_recommendation_notice: str = NO_ORCHESTRATOR_RECOMMENDATION_NOTICE
    scenario_id: str | None = None
    scenario_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", OrchestratorRunStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "updated_at", _aware_datetime("updated_at", self.updated_at))
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        object.__setattr__(
            self,
            "execution_policy",
            OrchestratorExecutionPolicy(self.execution_policy),
        )
        object.__setattr__(self, "plan", _plan_step_tuple(self.plan))
        object.__setattr__(
            self,
            "specialist_roles",
            _specialist_role_tuple(self.specialist_roles),
        )
        object.__setattr__(self, "agent_provider", _optional_text(self.agent_provider))
        object.__setattr__(self, "agent_model", _optional_text(self.agent_model))
        object.__setattr__(
            self,
            "orchestrator_skill_references",
            _text_tuple(
                "orchestrator_skill_references",
                self.orchestrator_skill_references,
            ),
        )
        if (self.agent_provider is None) != (self.agent_model is None):
            raise ValueError("agent_provider and agent_model must be provided together")
        object.__setattr__(self, "handoffs", _handoff_tuple(self.handoffs))
        object.__setattr__(
            self,
            "selected_company",
            _optional_object_mapping("selected_company", self.selected_company),
        )
        object.__setattr__(
            self,
            "selected_security",
            _optional_object_mapping("selected_security", self.selected_security),
        )
        object.__setattr__(self, "synthesis_summary", _optional_text(self.synthesis_summary))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(
            self,
            "no_recommendation_notice",
            _require_text("no_recommendation_notice", self.no_recommendation_notice),
        )
        object.__setattr__(self, "scenario_id", _optional_text(self.scenario_id))
        object.__setattr__(self, "scenario_version", _optional_text(self.scenario_version))
        if (self.scenario_id is None) != (self.scenario_version is None):
            raise ValueError("scenario_id and scenario_version must be provided together")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_policy": self.execution_policy.value,
            "plan": [step.to_dict() for step in self.plan],
            "specialist_roles": list(self.specialist_roles),
            "agent_provider": self.agent_provider,
            "agent_model": self.agent_model,
            "orchestrator_skill_references": list(self.orchestrator_skill_references),
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "selected_company": (
                dict(self.selected_company) if self.selected_company is not None else None
            ),
            "selected_security": (
                dict(self.selected_security) if self.selected_security is not None else None
            ),
            "synthesis_summary": self.synthesis_summary,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "no_recommendation_notice": self.no_recommendation_notice,
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
        }


def default_orchestrator_plan(
    specialist_roles: Iterable[str] = DEFAULT_RESEARCH_SPECIALIST_ROLES,
) -> tuple[OrchestratorPlanStep, ...]:
    roles = set(_specialist_role_tuple(specialist_roles))
    steps = (
        OrchestratorPlanStep(
            id="resolve_company",
            kind=OrchestratorStepKind.COMPANY_RESOLUTION,
            title="Resolve company and primary security",
            required=True,
        ),
        OrchestratorPlanStep(
            id="refresh_market_data",
            kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
            title="Refresh daily market data when a ticker is available",
            required=False,
        ),
        OrchestratorPlanStep(
            id="refresh_financial_statements",
            kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
            title="Refresh SEC companyfacts statements when a CIK is available",
            required=False,
        ),
        OrchestratorPlanStep(
            id="refresh_filings",
            kind=OrchestratorStepKind.FILING_REFRESH,
            title="Refresh latest SEC filings when a CIK is available",
            required=False,
        ),
        OrchestratorPlanStep(
            id="financial_report_analysis",
            kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            title="Run financial report specialist over stored statements and filings",
            required=False,
        ),
        OrchestratorPlanStep(
            id="stock_price_analysis",
            kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            title="Run stock price specialist over stored market data",
            required=False,
        ),
        OrchestratorPlanStep(
            id="context_analysis",
            kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
            title="Run news, macro, and sector context specialist over explicit sources",
            required=False,
        ),
        OrchestratorPlanStep(
            id="synthesis",
            kind=OrchestratorStepKind.SYNTHESIS,
            title="Create bounded synthesis from stored specialist outputs",
            required=True,
        ),
    )
    role_by_step = {
        OrchestratorStepKind.MARKET_DATA_REFRESH: "stock",
        OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH: "financial-report",
        OrchestratorStepKind.FILING_REFRESH: "financial-report",
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS: "financial-report",
        OrchestratorStepKind.STOCK_PRICE_ANALYSIS: "stock",
        OrchestratorStepKind.CONTEXT_ANALYSIS: "context",
        OrchestratorStepKind.SYNTHESIS: "synthesis",
    }
    return tuple(
        step
        for step in steps
        if (required_role := role_by_step.get(step.kind)) is None or required_role in roles
    )


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


def _optional_upper_text(value: str | None) -> str | None:
    text = _optional_text(value)
    return text.upper() if text is not None else None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _specialist_role_tuple(values: Iterable[str]) -> tuple[str, ...]:
    roles = tuple(dict.fromkeys(_text_tuple("specialist_roles", values)))
    unsupported = sorted(set(roles) - set(ALLOWED_RESEARCH_SPECIALIST_ROLES))
    if unsupported:
        raise ValueError(f"unsupported specialist role: {unsupported[0]}")
    if "synthesis" not in roles:
        raise ValueError("specialist_roles must include synthesis")
    return roles


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _positive_int_mapping(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, value in values.items():
        form = _require_text(f"{name}.key", key).upper()
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name}[{key!r}] must be a positive integer")
        normalized[form] = value
    return MappingProxyType(normalized)


def _object_mapping(name: str, values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {_require_text(f"{name}.key", key): value for key, value in values.items()}
    )


def _optional_object_mapping(
    name: str,
    values: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if values is None:
        return None
    return _object_mapping(name, values)


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _context_source_item_tuple(
    values: Iterable[ContextSourceItem],
) -> tuple[ContextSourceItem, ...]:
    items = tuple(values)
    for index, item in enumerate(items):
        if not isinstance(item, ContextSourceItem):
            raise ValueError(f"context_source_items[{index}] must be a ContextSourceItem")
    return items


def _plan_step_tuple(values: Iterable[OrchestratorPlanStep]) -> tuple[OrchestratorPlanStep, ...]:
    steps = tuple(values)
    for index, step in enumerate(steps):
        if not isinstance(step, OrchestratorPlanStep):
            raise ValueError(f"plan[{index}] must be an OrchestratorPlanStep")
    return steps


def _handoff_tuple(values: Iterable[AgentHandoff]) -> tuple[AgentHandoff, ...]:
    handoffs = tuple(values)
    for index, handoff in enumerate(handoffs):
        if not isinstance(handoff, AgentHandoff):
            raise ValueError(f"handoffs[{index}] must be an AgentHandoff")
    return handoffs
