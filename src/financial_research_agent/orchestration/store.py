from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Self

from financial_research_agent.orchestration.contracts import (
    AgentExecutionMetadata,
    AgentExecutionMode,
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorPlanStep,
    OrchestratorRunStatus,
    OrchestratorStepKind,
)
from financial_research_agent.settings import Settings

ORCHESTRATOR_RUN_STORE_VERSION = 1


class OrchestratorRunStore:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._runs: dict[str, OrchestratedResearchRun] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(storage_path=settings.local_paths.data_dir / "orchestrator_runs.json")

    def save(self, run: OrchestratedResearchRun) -> OrchestratedResearchRun:
        with self._lock:
            self._runs[run.id] = run
            self._save()
        return run

    def get(self, run_id: str) -> OrchestratedResearchRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> tuple[OrchestratedResearchRun, ...]:
        with self._lock:
            return tuple(sorted(self._runs.values(), key=lambda run: run.updated_at, reverse=True))

    def count(self) -> int:
        with self._lock:
            return len(self._runs)

    def clear(self) -> int:
        with self._lock:
            deleted = len(self._runs)
            self._runs.clear()
            self._save()
            return deleted

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != ORCHESTRATOR_RUN_STORE_VERSION:
                raise ValueError("unsupported orchestrator run store version")
            runs_payload = payload.get("runs", ())
            if not isinstance(runs_payload, list):
                raise ValueError("orchestrator runs must be a list")
            self._runs = {
                run.id: run
                for run in (orchestrated_research_run_from_dict(item) for item in runs_payload)
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"Could not load orchestrator run store: {self.storage_path}"
            raise ValueError(message) from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ORCHESTRATOR_RUN_STORE_VERSION,
            "runs": [run.to_dict() for run in self._runs.values()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def orchestrated_research_run_from_dict(payload: Any) -> OrchestratedResearchRun:
    if not isinstance(payload, dict):
        raise ValueError("orchestrator run must be an object")
    return OrchestratedResearchRun(
        id=str(payload["id"]),
        query=str(payload["query"]),
        status=OrchestratorRunStatus(str(payload["status"])),
        created_at=_datetime_from_payload(payload["created_at"]),
        updated_at=_datetime_from_payload(payload["updated_at"]),
        execution_policy=OrchestratorExecutionPolicy(str(payload["execution_policy"])),
        plan=tuple(_plan_step_from_payload(item) for item in payload.get("plan", ())),
        specialist_roles=tuple(
            str(item)
            for item in payload.get(
                "specialist_roles",
                ("financial-report", "stock", "context", "synthesis"),
            )
        ),
        agent_provider=_optional_payload_text(payload, "agent_provider"),
        agent_model=_optional_payload_text(payload, "agent_model"),
        orchestrator_skill_references=tuple(
            str(item) for item in payload.get("orchestrator_skill_references", ())
        ),
        handoffs=tuple(_handoff_from_payload(item) for item in payload.get("handoffs", ())),
        selected_company=_optional_mapping(payload.get("selected_company")),
        selected_security=_optional_mapping(payload.get("selected_security")),
        synthesis_summary=_optional_payload_text(payload, "synthesis_summary"),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        limitations=tuple(str(item) for item in payload.get("limitations", ())),
        no_recommendation_notice=str(payload["no_recommendation_notice"]),
        scenario_id=_optional_payload_text(payload, "scenario_id"),
        scenario_version=_optional_payload_text(payload, "scenario_version"),
    )


def _plan_step_from_payload(payload: Any) -> OrchestratorPlanStep:
    if not isinstance(payload, dict):
        raise ValueError("orchestrator plan step must be an object")
    return OrchestratorPlanStep(
        id=str(payload["id"]),
        kind=OrchestratorStepKind(str(payload["kind"])),
        title=str(payload["title"]),
        required=bool(payload["required"]),
        can_run_parallel=bool(payload.get("can_run_parallel", False)),
    )


def _handoff_from_payload(payload: Any) -> AgentHandoff:
    if not isinstance(payload, dict):
        raise ValueError("agent handoff must be an object")
    return AgentHandoff(
        id=str(payload["id"]),
        step_id=str(payload["step_id"]),
        kind=OrchestratorStepKind(str(payload["kind"])),
        status=OrchestratorHandoffStatus(str(payload["status"])),
        started_at=_datetime_from_payload(payload["started_at"]),
        completed_at=_datetime_from_payload(payload["completed_at"]),
        input_summary=_str_mapping(payload.get("input_summary", {})),
        output=_object_mapping(payload.get("output", {})),
        evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        limitations=tuple(str(item) for item in payload.get("limitations", ())),
        confidence=HandoffConfidence(str(payload.get("confidence", HandoffConfidence.UNKNOWN))),
        error_code=_optional_payload_text(payload, "error_code"),
        error_message=_optional_payload_text(payload, "error_message"),
        execution=_execution_from_payload(payload.get("execution")),
    )


def handoff_from_dict(payload: dict[str, object]) -> AgentHandoff:
    return _handoff_from_payload(payload)


def _execution_from_payload(value: Any) -> AgentExecutionMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("execution metadata must be an object")
    return AgentExecutionMetadata(
        mode=AgentExecutionMode(str(value["mode"])),
        agent_role=str(value["agent_role"]),
        correlation_id=str(value["correlation_id"]),
        delegation_id=_optional_payload_text(value, "delegation_id"),
        remote_task_id=_optional_payload_text(value, "remote_task_id"),
        service_id=_optional_payload_text(value, "service_id"),
        attempt_count=int(value.get("attempt_count", 1)),
        prompt_id=_optional_payload_text(value, "prompt_id"),
        prompt_version=_optional_payload_text(value, "prompt_version"),
        provider=_optional_payload_text(value, "provider"),
        model=_optional_payload_text(value, "model"),
        tool_status=_optional_payload_text(value, "tool_status"),
        reasoning_summary=_optional_payload_text(value, "reasoning_summary"),
        skill_references=tuple(str(item) for item in value.get("skill_references", ())),
    )


def _datetime_from_payload(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _optional_payload_text(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    return str(value)


def _optional_mapping(value: Any) -> dict[str, object] | None:
    if value is None:
        return None
    return _object_mapping(value)


def _str_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("mapping payload must be an object")
    return {str(key): str(item) for key, item in value.items()}


def _object_mapping(value: Any) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("mapping payload must be an object")
    return {str(key): item for key, item in value.items()}
