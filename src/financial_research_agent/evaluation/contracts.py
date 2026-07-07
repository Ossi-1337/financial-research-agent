from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

EVALUATION_SCHEMA_VERSION = 1


class EvalArtifactKind(StrEnum):
    CITED_ANSWER = "cited_answer"
    SYNTHESIS_REPORT = "synthesis_report"
    ORCHESTRATOR_RUN = "orchestrator_run"
    GENERIC = "generic"


class EvalDatasetLabel(StrEnum):
    REAL = "real"
    FIXTURE = "fixture"


class EvalCheckKind(StrEnum):
    SCHEMA_VALIDITY = "schema_validity"
    CITATION_COVERAGE = "citation_coverage"
    SOURCE_FRESHNESS = "source_freshness"
    REFUSAL_BEHAVIOR = "refusal_behavior"
    HALLUCINATION_GUARDRAIL = "hallucination_guardrail"
    TRACEABILITY = "traceability"
    LLM_JUDGE = "llm_judge"


class EvalCheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class EvalSuiteStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    query: str
    artifact_kind: EvalArtifactKind
    dataset_label: EvalDatasetLabel
    description: str
    required_schema_paths: tuple[str, ...] = ()
    min_citations: int = 0
    required_citation_markers: tuple[str, ...] = ()
    required_evidence_ids: tuple[str, ...] = ()
    max_source_age_days: int | None = None
    expects_refusal: bool = False
    required_refusal_terms: tuple[str, ...] = ()
    forbidden_claim_patterns: tuple[str, ...] = ()
    required_trace_components: tuple[str, ...] = ()
    llm_judge_prompt_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "artifact_kind", EvalArtifactKind(self.artifact_kind))
        object.__setattr__(self, "dataset_label", EvalDatasetLabel(self.dataset_label))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(
            self,
            "required_schema_paths",
            _text_tuple("required_schema_paths", self.required_schema_paths),
        )
        if self.min_citations < 0:
            raise ValueError("min_citations must be non-negative")
        if self.max_source_age_days is not None and self.max_source_age_days <= 0:
            raise ValueError("max_source_age_days must be positive")
        object.__setattr__(
            self,
            "required_citation_markers",
            _text_tuple("required_citation_markers", self.required_citation_markers),
        )
        object.__setattr__(
            self,
            "required_evidence_ids",
            _text_tuple("required_evidence_ids", self.required_evidence_ids),
        )
        object.__setattr__(
            self,
            "required_refusal_terms",
            _text_tuple("required_refusal_terms", self.required_refusal_terms),
        )
        object.__setattr__(
            self,
            "forbidden_claim_patterns",
            _text_tuple("forbidden_claim_patterns", self.forbidden_claim_patterns),
        )
        object.__setattr__(
            self,
            "required_trace_components",
            _text_tuple("required_trace_components", self.required_trace_components),
        )
        object.__setattr__(
            self,
            "llm_judge_prompt_id",
            _optional_text(self.llm_judge_prompt_id),
        )
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "artifact_kind": self.artifact_kind.value,
            "dataset_label": self.dataset_label.value,
            "description": self.description,
            "required_schema_paths": list(self.required_schema_paths),
            "min_citations": self.min_citations,
            "required_citation_markers": list(self.required_citation_markers),
            "required_evidence_ids": list(self.required_evidence_ids),
            "max_source_age_days": self.max_source_age_days,
            "expects_refusal": self.expects_refusal,
            "required_refusal_terms": list(self.required_refusal_terms),
            "forbidden_claim_patterns": list(self.forbidden_claim_patterns),
            "required_trace_components": list(self.required_trace_components),
            "llm_judge_prompt_id": self.llm_judge_prompt_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvalArtifact:
    case_id: str
    artifact_kind: EvalArtifactKind
    payload: Mapping[str, object]
    trace: Mapping[str, object] = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_text("case_id", self.case_id))
        object.__setattr__(self, "artifact_kind", EvalArtifactKind(self.artifact_kind))
        object.__setattr__(self, "payload", _object_mapping("payload", self.payload))
        object.__setattr__(self, "trace", _object_mapping("trace", self.trace))
        object.__setattr__(self, "provider", _optional_text(self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "artifact_kind": self.artifact_kind.value,
            "payload": dict(self.payload),
            "trace": dict(self.trace),
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class EvalCheckResult:
    case_id: str
    kind: EvalCheckKind
    status: EvalCheckStatus
    message: str
    component: str | None = None
    source: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_text("case_id", self.case_id))
        object.__setattr__(self, "kind", EvalCheckKind(self.kind))
        object.__setattr__(self, "status", EvalCheckStatus(self.status))
        object.__setattr__(self, "message", _require_text("message", self.message))
        object.__setattr__(self, "component", _optional_text(self.component))
        object.__setattr__(self, "source", _optional_text(self.source))
        object.__setattr__(self, "details", _object_mapping("details", self.details))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "message": self.message,
            "component": self.component,
            "source": self.source,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    case: EvalCase
    checks: tuple[EvalCheckResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, EvalCase):
            raise ValueError("case must be an EvalCase")
        object.__setattr__(self, "checks", _check_tuple(self.checks))

    @property
    def status(self) -> EvalSuiteStatus:
        if any(check.status == EvalCheckStatus.FAILED for check in self.checks):
            return EvalSuiteStatus.FAILED
        return EvalSuiteStatus.PASSED

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case.to_dict(),
            "status": self.status.value,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class EvalSuiteResult:
    id: str
    generated_at: datetime
    case_results: tuple[EvalCaseResult, ...]
    schema_version: int = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "generated_at", _aware_datetime("generated_at", self.generated_at))
        object.__setattr__(self, "case_results", _case_result_tuple(self.case_results))
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")

    @property
    def status(self) -> EvalSuiteStatus:
        if any(result.status == EvalSuiteStatus.FAILED for result in self.case_results):
            return EvalSuiteStatus.FAILED
        return EvalSuiteStatus.PASSED

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.case_results if result.status == EvalSuiteStatus.FAILED)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.case_results if result.status == EvalSuiteStatus.PASSED)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "case_count": len(self.case_results),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "case_results": [result.to_dict() for result in self.case_results],
        }


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


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", str(key)): _require_text(f"{name}[{key!r}]", str(value))
            for key, value in values.items()
        }
    )


def _object_mapping(name: str, values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {_require_text(f"{name}.key", str(key)): value for key, value in values.items()}
    )


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _check_tuple(values: Iterable[EvalCheckResult]) -> tuple[EvalCheckResult, ...]:
    checks = tuple(values)
    for index, check in enumerate(checks):
        if not isinstance(check, EvalCheckResult):
            raise ValueError(f"checks[{index}] must be an EvalCheckResult")
    return checks


def _case_result_tuple(values: Iterable[EvalCaseResult]) -> tuple[EvalCaseResult, ...]:
    results = tuple(values)
    for index, result in enumerate(results):
        if not isinstance(result, EvalCaseResult):
            raise ValueError(f"case_results[{index}] must be an EvalCaseResult")
    return results
