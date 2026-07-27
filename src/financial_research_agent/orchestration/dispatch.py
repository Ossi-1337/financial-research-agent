from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from financial_research_agent.orchestration.contracts import (
    AgentHandoff,
    OrchestratedResearchRun,
    OrchestratorStepKind,
)


class AgentRole(StrEnum):
    COMPANY_RESEARCH = "company-research"
    FINANCIAL_REPORT = "financial-report"
    STOCK = "stock"
    CONTEXT = "context"
    SYNTHESIS = "synthesis"


class AgentTopologyMode(StrEnum):
    SINGLE = "single"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True, slots=True)
class AgentEndpoint:
    role: AgentRole
    service_id: str
    base_url: str
    skill_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", AgentRole(self.role))
        object.__setattr__(self, "service_id", _required_text("service_id", self.service_id))
        base_url = _required_text("base_url", self.base_url).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "skill_id", _required_text("skill_id", self.skill_id))


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    role: AgentRole
    run_id: str
    step_id: str
    correlation_id: str
    expected_kind: OrchestratorStepKind
    payload: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", AgentRole(self.role))
        object.__setattr__(self, "run_id", _required_text("run_id", self.run_id))
        object.__setattr__(self, "step_id", _required_text("step_id", self.step_id))
        object.__setattr__(
            self,
            "correlation_id",
            _required_text("correlation_id", self.correlation_id),
        )
        object.__setattr__(self, "expected_kind", OrchestratorStepKind(self.expected_kind))
        if self.schema_version != 1:
            raise ValueError("unsupported delegation schema_version")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role.value,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "correlation_id": self.correlation_id,
            "expected_kind": self.expected_kind.value,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class DelegationResult:
    handoff: AgentHandoff
    remote_task_id: str | None = None
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.handoff, AgentHandoff):
            raise ValueError("handoff must be an AgentHandoff")
        object.__setattr__(self, "remote_task_id", _optional_text(self.remote_task_id))
        if self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")


class ResearchStepDispatcher(Protocol):
    async def dispatch(
        self,
        request: DelegationRequest,
        *,
        run: OrchestratedResearchRun | None = None,
    ) -> DelegationResult: ...


LocalDispatchHandler = Callable[
    [DelegationRequest, OrchestratedResearchRun | None],
    Awaitable[AgentHandoff],
]


class LocalResearchStepDispatcher:
    def __init__(self, handler: LocalDispatchHandler) -> None:
        self._handler = handler

    async def dispatch(
        self,
        request: DelegationRequest,
        *,
        run: OrchestratedResearchRun | None = None,
    ) -> DelegationResult:
        return DelegationResult(handoff=await self._handler(request, run))


def delegation_request_from_dict(payload: Mapping[str, object]) -> DelegationRequest:
    raw_payload = payload.get("payload", {})
    if not isinstance(raw_payload, Mapping):
        raise ValueError("payload must be an object")
    return DelegationRequest(
        schema_version=int(payload.get("schema_version", 0)),
        role=AgentRole(str(payload["role"])),
        run_id=str(payload["run_id"]),
        step_id=str(payload["step_id"]),
        correlation_id=str(payload["correlation_id"]),
        expected_kind=OrchestratorStepKind(str(payload["expected_kind"])),
        payload={str(key): value for key, value in raw_payload.items()},
    )


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None
