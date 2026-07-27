from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from financial_research_agent import __version__
from financial_research_agent.orchestration import (
    AgentHandoff,
    OrchestratedResearchRun,
    OrchestratorStepKind,
)
from financial_research_agent.settings import Settings

OBSERVABILITY_SCHEMA_VERSION = 1
DEFAULT_TEXT_PREVIEW_CHARS = 1_200
DEFAULT_COLLECTION_PREVIEW_ITEMS = 8


class TraceEventKind(StrEnum):
    PROVIDER_CALL = "provider_call"
    TOOL_CALL = "tool_call"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
    AGENT_OUTPUT = "agent_output"
    WARNING = "warning"
    TOKEN_COST_ESTIMATE = "token_cost_estimate"
    REPLAY_STEP = "replay_step"


class ReplayMode(StrEnum):
    STORED_RESULT = "stored_result"
    NOT_REPLAYABLE = "not_replayable"


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    sensitive_values: tuple[str, ...] = ()
    sensitive_paths: tuple[str, ...] = ()
    text_preview_chars: int = DEFAULT_TEXT_PREVIEW_CHARS
    collection_preview_items: int = DEFAULT_COLLECTION_PREVIEW_ITEMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitive_values", _text_tuple(self.sensitive_values))
        object.__setattr__(self, "sensitive_paths", _text_tuple(self.sensitive_paths))
        if self.text_preview_chars <= 0:
            raise ValueError("text_preview_chars must be positive")
        if self.collection_preview_items <= 0:
            raise ValueError("collection_preview_items must be positive")

    @classmethod
    def from_settings(cls, settings: Settings) -> RedactionPolicy:
        sensitive_values = tuple(
            value
            for value in (
                settings.provider.openai_api_key,
                settings.provider.anthropic_api_key,
                settings.provider.gemini_api_key,
                settings.provider.litellm_api_key,
                settings.data_sources.alpha_vantage_api_key,
                settings.a2a.api_key,
            )
            if value
        )
        sensitive_paths = (
            str(settings.local_paths.app_home),
            str(settings.local_paths.cache_dir),
            str(Path.home() / ".cache" / "huggingface"),
        )
        return cls(sensitive_values=sensitive_values, sensitive_paths=sensitive_paths)

    def redact(self, value: Any) -> Any:
        return _redact_value(value, self)

    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "sensitive_value_count": len(self.sensitive_values),
                "sensitive_path_count": len(self.sensitive_paths),
                "text_preview_chars": self.text_preview_chars,
                "collection_preview_items": self.collection_preview_items,
            }
        )


@dataclass(frozen=True, slots=True)
class TokenCostEstimate:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: str | None = None
    source: str = "not_reported"

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "estimated_cost_usd", _optional_text(self.estimated_cost_usd))
        object.__setattr__(self, "source", _require_text("source", self.source))

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TraceEvent:
    id: str
    run_id: str
    sequence: int
    kind: TraceEventKind
    title: str
    component: str
    status: str
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    safe_input: Mapping[str, object] = field(default_factory=dict)
    safe_output: Mapping[str, object] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    token_cost_estimate: TokenCostEstimate = field(default_factory=TokenCostEstimate)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "run_id", _require_text("run_id", self.run_id))
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        object.__setattr__(self, "kind", TraceEventKind(self.kind))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "component", _require_text("component", self.component))
        object.__setattr__(self, "status", _require_text("status", self.status))
        object.__setattr__(self, "started_at", _aware_datetime("started_at", self.started_at))
        object.__setattr__(self, "completed_at", _aware_datetime("completed_at", self.completed_at))
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must be greater than or equal to started_at")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        object.__setattr__(self, "safe_input", _object_mapping("safe_input", self.safe_input))
        object.__setattr__(self, "safe_output", _object_mapping("safe_output", self.safe_output))
        object.__setattr__(self, "evidence_ids", _text_tuple(self.evidence_ids))
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))
        object.__setattr__(self, "limitations", _text_tuple(self.limitations))
        object.__setattr__(self, "error_code", _optional_text(self.error_code))
        object.__setattr__(self, "error_message", _optional_text(self.error_message))
        if not isinstance(self.token_cost_estimate, TokenCostEstimate):
            raise ValueError("token_cost_estimate must be a TokenCostEstimate")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "title": self.title,
            "component": self.component,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "safe_input": dict(self.safe_input),
            "safe_output": dict(self.safe_output),
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "token_cost_estimate": self.token_cost_estimate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResearchTrace:
    run_id: str
    query: str
    status: str
    created_at: datetime
    updated_at: datetime
    events: tuple[TraceEvent, ...]
    redaction: Mapping[str, object]
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text("run_id", self.run_id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", _require_text("status", self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "updated_at", _aware_datetime("updated_at", self.updated_at))
        object.__setattr__(self, "events", _event_tuple(self.events))
        object.__setattr__(self, "redaction", _object_mapping("redaction", self.redaction))
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))
        object.__setattr__(self, "limitations", _text_tuple(self.limitations))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "events": [event.to_dict() for event in self.events],
            "redaction": dict(self.redaction),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ReplayStep:
    sequence: int
    step_id: str
    component: str
    mode: ReplayMode
    status: str
    reason: str
    safe_input: Mapping[str, object] = field(default_factory=dict)
    safe_output: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        object.__setattr__(self, "step_id", _require_text("step_id", self.step_id))
        object.__setattr__(self, "component", _require_text("component", self.component))
        object.__setattr__(self, "mode", ReplayMode(self.mode))
        object.__setattr__(self, "status", _require_text("status", self.status))
        object.__setattr__(self, "reason", _require_text("reason", self.reason))
        object.__setattr__(self, "safe_input", _object_mapping("safe_input", self.safe_input))
        object.__setattr__(self, "safe_output", _object_mapping("safe_output", self.safe_output))

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "step_id": self.step_id,
            "component": self.component,
            "mode": self.mode.value,
            "status": self.status,
            "reason": self.reason,
            "safe_input": dict(self.safe_input),
            "safe_output": dict(self.safe_output),
        }


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    run_id: str
    replayable: bool
    steps: tuple[ReplayStep, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text("run_id", self.run_id))
        object.__setattr__(self, "steps", _replay_step_tuple(self.steps))
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "replayable": self.replayable,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DebugBundle:
    schema_version: int
    created_at: datetime
    app_version: str
    run: Mapping[str, object]
    trace: ResearchTrace
    replay: ReplayPlan
    excluded_items: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "app_version", _require_text("app_version", self.app_version))
        object.__setattr__(self, "run", _object_mapping("run", self.run))
        if not isinstance(self.trace, ResearchTrace):
            raise ValueError("trace must be a ResearchTrace")
        if not isinstance(self.replay, ReplayPlan):
            raise ValueError("replay must be a ReplayPlan")
        object.__setattr__(self, "excluded_items", _text_tuple(self.excluded_items))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "app_version": self.app_version,
            "run": dict(self.run),
            "trace": self.trace.to_dict(),
            "replay": self.replay.to_dict(),
            "excluded_items": list(self.excluded_items),
        }


def build_trace_from_orchestrator_run(
    run: OrchestratedResearchRun,
    *,
    redaction_policy: RedactionPolicy | None = None,
) -> ResearchTrace:
    policy = redaction_policy or RedactionPolicy()
    events = tuple(
        _event_from_handoff(
            run_id=run.id,
            sequence=index,
            handoff=handoff,
            policy=policy,
        )
        for index, handoff in enumerate(run.handoffs, start=1)
    )
    return ResearchTrace(
        run_id=run.id,
        query=policy.redact(run.query),
        status=run.status.value,
        created_at=run.created_at,
        updated_at=run.updated_at,
        events=events,
        redaction=policy.metadata(),
        warnings=tuple(str(policy.redact(warning)) for warning in run.warnings),
        limitations=tuple(str(policy.redact(limitation)) for limitation in run.limitations),
    )


def build_replay_plan(
    run: OrchestratedResearchRun,
    *,
    redaction_policy: RedactionPolicy | None = None,
) -> ReplayPlan:
    policy = redaction_policy or RedactionPolicy()
    steps = tuple(
        _replay_step_from_handoff(index=index, handoff=handoff, policy=policy)
        for index, handoff in enumerate(run.handoffs, start=1)
    )
    replayable = bool(steps) and any(step.mode == ReplayMode.STORED_RESULT for step in steps)
    warnings = (
        "Replay uses stored deterministic handoff results only; it does not call providers, "
        "LLMs, tools, or data sources."
    )
    return ReplayPlan(run_id=run.id, replayable=replayable, steps=steps, warnings=(warnings,))


def build_debug_bundle(
    run: OrchestratedResearchRun,
    *,
    settings: Settings | None = None,
    redaction_policy: RedactionPolicy | None = None,
    created_at: datetime | None = None,
) -> DebugBundle:
    policy = redaction_policy or (
        RedactionPolicy.from_settings(settings) if settings is not None else RedactionPolicy()
    )
    trace = build_trace_from_orchestrator_run(run, redaction_policy=policy)
    replay = build_replay_plan(run, redaction_policy=policy)
    return DebugBundle(
        schema_version=OBSERVABILITY_SCHEMA_VERSION,
        created_at=created_at or datetime.now(UTC),
        app_version=__version__,
        run=policy.redact(run.to_dict()),
        trace=trace,
        replay=replay,
        excluded_items=(
            "raw provider credentials",
            "model cache files",
            "raw local documents outside stored run payloads",
            "hidden chain-of-thought logs",
            "hosted telemetry",
        ),
    )


def _event_from_handoff(
    *,
    run_id: str,
    sequence: int,
    handoff: AgentHandoff,
    policy: RedactionPolicy,
) -> TraceEvent:
    return TraceEvent(
        id=f"trace_event_{handoff.step_id}_{sequence}",
        run_id=run_id,
        sequence=sequence,
        kind=_event_kind(handoff.kind),
        title=handoff.step_id.replace("_", " ").title(),
        component=handoff.kind.value,
        status=handoff.status.value,
        started_at=handoff.started_at,
        completed_at=handoff.completed_at,
        duration_ms=_duration_ms(handoff.started_at, handoff.completed_at),
        safe_input=_mapping_or_empty(
            policy.redact(
                {
                    **dict(handoff.input_summary),
                    **(
                        {"execution": handoff.execution.to_dict()}
                        if handoff.execution is not None
                        else {}
                    ),
                }
            )
        ),
        safe_output=_mapping_or_empty(policy.redact(handoff.output)),
        evidence_ids=tuple(str(policy.redact(item)) for item in handoff.evidence_ids),
        warnings=tuple(str(policy.redact(item)) for item in handoff.warnings),
        limitations=tuple(str(policy.redact(item)) for item in handoff.limitations),
        error_code=handoff.error_code,
        error_message=(
            str(policy.redact(handoff.error_message)) if handoff.error_message is not None else None
        ),
        token_cost_estimate=_token_estimate_from_output(handoff.output),
    )


def _replay_step_from_handoff(
    *,
    index: int,
    handoff: AgentHandoff,
    policy: RedactionPolicy,
) -> ReplayStep:
    has_stored_output = bool(handoff.output) or handoff.error_message is not None
    mode = ReplayMode.STORED_RESULT if has_stored_output else ReplayMode.NOT_REPLAYABLE
    reason = (
        "Stored handoff output can be replayed without provider calls."
        if has_stored_output
        else "No stored output is available for this step."
    )
    return ReplayStep(
        sequence=index,
        step_id=handoff.step_id,
        component=handoff.kind.value,
        mode=mode,
        status=handoff.status.value,
        reason=reason,
        safe_input=_mapping_or_empty(policy.redact(handoff.input_summary)),
        safe_output=_mapping_or_empty(policy.redact(_replay_output(handoff))),
    )


def _event_kind(kind: OrchestratorStepKind) -> TraceEventKind:
    if kind in {
        OrchestratorStepKind.COMPANY_RESOLUTION,
        OrchestratorStepKind.MARKET_DATA_REFRESH,
        OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
        OrchestratorStepKind.FILING_REFRESH,
    }:
        return TraceEventKind.PROVIDER_CALL
    if kind in {
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        OrchestratorStepKind.CONTEXT_ANALYSIS,
        OrchestratorStepKind.SYNTHESIS,
    }:
        return TraceEventKind.AGENT_OUTPUT
    return TraceEventKind.REPLAY_STEP


def _replay_output(handoff: AgentHandoff) -> Mapping[str, object]:
    output = dict(handoff.output)
    if handoff.error_code is not None:
        output["error_code"] = handoff.error_code
    if handoff.error_message is not None:
        output["error_message"] = handoff.error_message
    return output


def _token_estimate_from_output(output: Mapping[str, object]) -> TokenCostEstimate:
    usage = _find_usage(output)
    if usage is None:
        return TokenCostEstimate(source="not_reported")
    return TokenCostEstimate(
        input_tokens=_optional_int(usage.get("input_tokens")),
        output_tokens=_optional_int(usage.get("output_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        source="provider_usage",
    )


def _find_usage(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        if {"input_tokens", "output_tokens", "total_tokens"} & set(value.keys()):
            return value
        for item in value.values():
            result = _find_usage(item)
            if result is not None:
                return result
    if isinstance(value, list | tuple):
        for item in value:
            result = _find_usage(item)
            if result is not None:
                return result
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
    return max(0, int((completed_at - started_at).total_seconds() * 1000))


def _redact_value(value: Any, policy: RedactionPolicy) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = (
                "[REDACTED]" if _is_sensitive_key(key_text) else _redact_value(item, policy)
            )
        return _preview_mapping(redacted, policy.collection_preview_items)
    if isinstance(value, list | tuple):
        return [
            _redact_value(item, policy) for item in list(value)[: policy.collection_preview_items]
        ] + _remaining_marker(len(value), policy.collection_preview_items)
    if isinstance(value, str):
        return _redact_text(value, policy)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _redact_text(str(value), policy)


def _redact_text(value: str, policy: RedactionPolicy) -> str:
    text = value
    for secret in policy.sensitive_values:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for path in policy.sensitive_paths:
        if path:
            text = text.replace(path, "[LOCAL_PATH]")
            text = text.replace(path.replace("\\", "/"), "[LOCAL_PATH]")
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(sk-[A-Za-z0-9_\-]{8,})", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^,\s;]+",
        r"\1=[REDACTED]",
        text,
    )
    if len(text) > policy.text_preview_chars:
        return f"{text[: policy.text_preview_chars]}... [truncated]"
    return text


def _preview_mapping(values: Mapping[str, object], limit: int) -> dict[str, object]:
    items = list(values.items())
    preview = dict(items[:limit])
    if len(items) > limit:
        preview["_truncated_items"] = len(items) - limit
    return preview


def _remaining_marker(length: int, limit: int) -> list[str]:
    remaining = length - limit
    return [f"[{remaining} more items truncated]"] if remaining > 0 else []


def _is_sensitive_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    if "api_key" in normalized or "apikey" in normalized:
        return True
    if "secret" in normalized or "password" in normalized:
        return True
    if normalized in {"authorization", "auth", "token"}:
        return True
    return normalized.endswith("_token")


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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


def _text_tuple(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("value must be an iterable of strings, not a string")
    return tuple(_require_text(f"value[{index}]", value) for index, value in enumerate(values))


def _object_mapping(name: str, values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {_require_text(f"{name}.key", str(key)): item for key, item in values.items()}
    )


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _event_tuple(values: Iterable[TraceEvent]) -> tuple[TraceEvent, ...]:
    events = tuple(values)
    for index, event in enumerate(events):
        if not isinstance(event, TraceEvent):
            raise ValueError(f"events[{index}] must be a TraceEvent")
    return events


def _replay_step_tuple(values: Iterable[ReplayStep]) -> tuple[ReplayStep, ...]:
    steps = tuple(values)
    for index, step in enumerate(steps):
        if not isinstance(step, ReplayStep):
            raise ValueError(f"steps[{index}] must be a ReplayStep")
    return steps


def to_pretty_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
