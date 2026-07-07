from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, time

from financial_research_agent.evaluation.contracts import (
    EVALUATION_SCHEMA_VERSION,
    EvalArtifact,
    EvalCase,
    EvalCaseResult,
    EvalCheckKind,
    EvalCheckResult,
    EvalCheckStatus,
    EvalSuiteResult,
)
from financial_research_agent.evaluation.dataset import (
    DEFAULT_EVALUATION_SUITE_ID,
    default_eval_artifacts,
    default_eval_cases,
)


def run_default_offline_evaluations(
    *,
    now: datetime | None = None,
) -> EvalSuiteResult:
    generated_at = now or datetime.now(UTC)
    return run_evaluation_suite(
        suite_id=DEFAULT_EVALUATION_SUITE_ID,
        cases=default_eval_cases(),
        artifacts=default_eval_artifacts(now=generated_at),
        now=generated_at,
    )


def run_evaluation_suite(
    *,
    suite_id: str,
    cases: Iterable[EvalCase],
    artifacts: Iterable[EvalArtifact],
    now: datetime | None = None,
) -> EvalSuiteResult:
    generated_at = now or datetime.now(UTC)
    artifacts_by_case = {artifact.case_id: artifact for artifact in artifacts}
    case_results = tuple(
        evaluate_case(
            case=case,
            artifact=artifacts_by_case.get(case.id),
            now=generated_at,
        )
        for case in cases
    )
    return EvalSuiteResult(
        id=suite_id,
        generated_at=generated_at,
        case_results=case_results,
        schema_version=EVALUATION_SCHEMA_VERSION,
    )


def evaluate_case(
    *,
    case: EvalCase,
    artifact: EvalArtifact | None,
    now: datetime | None = None,
) -> EvalCaseResult:
    generated_at = now or datetime.now(UTC)
    if artifact is None:
        return EvalCaseResult(
            case=case,
            checks=(
                _result(
                    case,
                    EvalCheckKind.SCHEMA_VALIDITY,
                    EvalCheckStatus.FAILED,
                    "No artifact was provided for this eval case.",
                    component=case.artifact_kind.value,
                ),
            ),
        )
    if artifact.artifact_kind != case.artifact_kind:
        return EvalCaseResult(
            case=case,
            checks=(
                _result(
                    case,
                    EvalCheckKind.SCHEMA_VALIDITY,
                    EvalCheckStatus.FAILED,
                    "Artifact kind does not match the eval case.",
                    component=case.artifact_kind.value,
                    details={
                        "expected": case.artifact_kind.value,
                        "actual": artifact.artifact_kind.value,
                    },
                ),
            ),
        )

    checks = [
        _check_schema(case, artifact),
        _check_citations(case, artifact),
        _check_source_freshness(case, artifact, now=generated_at),
        _check_refusal(case, artifact),
        _check_hallucination_sensitive_claims(case, artifact),
        _check_traceability(case, artifact),
        _check_llm_judge(case),
    ]
    return EvalCaseResult(case=case, checks=tuple(checks))


def _check_schema(case: EvalCase, artifact: EvalArtifact) -> EvalCheckResult:
    missing = [
        path for path in case.required_schema_paths if not _path_has_value(artifact.payload, path)
    ]
    if missing:
        return _result(
            case,
            EvalCheckKind.SCHEMA_VALIDITY,
            EvalCheckStatus.FAILED,
            "Required schema paths are missing or empty.",
            component=case.artifact_kind.value,
            details={"missing_paths": missing},
        )
    return _result(
        case,
        EvalCheckKind.SCHEMA_VALIDITY,
        EvalCheckStatus.PASSED,
        "Required schema paths are present.",
        component=case.artifact_kind.value,
        details={"checked_paths": list(case.required_schema_paths)},
    )


def _check_citations(case: EvalCase, artifact: EvalArtifact) -> EvalCheckResult:
    citations = _mapping_list(artifact.payload.get("citations", ()))
    citation_markers = {
        str(citation.get("marker") or f"[{citation.get('id')}]") for citation in citations
    }
    evidence_ids = _collect_values_for_keys(artifact.payload, {"evidence_id", "evidence_ids"})
    missing_markers = sorted(set(case.required_citation_markers) - citation_markers)
    missing_evidence = sorted(set(case.required_evidence_ids) - evidence_ids)

    if len(citations) < case.min_citations or missing_markers or missing_evidence:
        return _result(
            case,
            EvalCheckKind.CITATION_COVERAGE,
            EvalCheckStatus.FAILED,
            "Citation or evidence requirements were not met.",
            component="retrieval",
            source=_first_source_url(citations),
            details={
                "citation_count": len(citations),
                "min_citations": case.min_citations,
                "missing_markers": missing_markers,
                "missing_evidence_ids": missing_evidence,
            },
        )
    return _result(
        case,
        EvalCheckKind.CITATION_COVERAGE,
        EvalCheckStatus.PASSED,
        "Citation and evidence requirements were met.",
        component="retrieval",
        source=_first_source_url(citations),
        details={"citation_count": len(citations), "evidence_ids": sorted(evidence_ids)},
    )


def _check_source_freshness(
    case: EvalCase,
    artifact: EvalArtifact,
    *,
    now: datetime,
) -> EvalCheckResult:
    if case.max_source_age_days is None:
        return _result(
            case,
            EvalCheckKind.SOURCE_FRESHNESS,
            EvalCheckStatus.SKIPPED,
            "No source freshness requirement is configured for this case.",
            component="source_freshness",
        )

    timestamps = _collect_datetime_values(
        artifact.payload,
        {"retrieved_at", "published_at", "data_as_of"},
    )
    if not timestamps:
        return _result(
            case,
            EvalCheckKind.SOURCE_FRESHNESS,
            EvalCheckStatus.FAILED,
            "No source freshness timestamp was found.",
            component="source_freshness",
        )
    oldest = min(timestamps)
    age_days = (now - oldest).days
    if age_days > case.max_source_age_days:
        return _result(
            case,
            EvalCheckKind.SOURCE_FRESHNESS,
            EvalCheckStatus.FAILED,
            "At least one source timestamp is older than allowed.",
            component="source_freshness",
            details={"oldest": oldest.isoformat(), "age_days": age_days},
        )
    return _result(
        case,
        EvalCheckKind.SOURCE_FRESHNESS,
        EvalCheckStatus.PASSED,
        "Source freshness requirement was met.",
        component="source_freshness",
        details={"oldest": oldest.isoformat(), "age_days": age_days},
    )


def _check_refusal(case: EvalCase, artifact: EvalArtifact) -> EvalCheckResult:
    if not case.expects_refusal:
        return _result(
            case,
            EvalCheckKind.REFUSAL_BEHAVIOR,
            EvalCheckStatus.SKIPPED,
            "This case does not require refusal behavior.",
            component="refusal",
        )
    text = _text_blob(artifact.payload).casefold()
    missing_terms = [term for term in case.required_refusal_terms if term.casefold() not in text]
    if missing_terms:
        return _result(
            case,
            EvalCheckKind.REFUSAL_BEHAVIOR,
            EvalCheckStatus.FAILED,
            "Expected refusal language was not found.",
            component="refusal",
            details={"missing_terms": missing_terms},
        )
    return _result(
        case,
        EvalCheckKind.REFUSAL_BEHAVIOR,
        EvalCheckStatus.PASSED,
        "Expected refusal language was found.",
        component="refusal",
    )


def _check_hallucination_sensitive_claims(
    case: EvalCase,
    artifact: EvalArtifact,
) -> EvalCheckResult:
    text = _text_blob(artifact.payload)
    matches = [
        pattern
        for pattern in case.forbidden_claim_patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    if matches:
        return _result(
            case,
            EvalCheckKind.HALLUCINATION_GUARDRAIL,
            EvalCheckStatus.FAILED,
            "Forbidden hallucination-sensitive claim pattern was found.",
            component=case.artifact_kind.value,
            details={"matched_patterns": matches},
        )
    return _result(
        case,
        EvalCheckKind.HALLUCINATION_GUARDRAIL,
        EvalCheckStatus.PASSED,
        "No forbidden hallucination-sensitive claim patterns were found.",
        component=case.artifact_kind.value,
        details={"checked_patterns": list(case.forbidden_claim_patterns)},
    )


def _check_traceability(case: EvalCase, artifact: EvalArtifact) -> EvalCheckResult:
    if not case.required_trace_components:
        return _result(
            case,
            EvalCheckKind.TRACEABILITY,
            EvalCheckStatus.SKIPPED,
            "This case does not require trace components.",
            component="trace",
        )
    events = _mapping_list(artifact.trace.get("events", ()))
    components = {str(event.get("component")) for event in events if event.get("component")}
    missing = sorted(set(case.required_trace_components) - components)
    if missing:
        return _result(
            case,
            EvalCheckKind.TRACEABILITY,
            EvalCheckStatus.FAILED,
            "Required trace components were not found.",
            component="trace",
            details={"missing_components": missing, "components": sorted(components)},
        )
    return _result(
        case,
        EvalCheckKind.TRACEABILITY,
        EvalCheckStatus.PASSED,
        "Required trace components were found.",
        component="trace",
        details={"components": sorted(components)},
    )


def _check_llm_judge(case: EvalCase) -> EvalCheckResult:
    if case.llm_judge_prompt_id is None:
        return _result(
            case,
            EvalCheckKind.LLM_JUDGE,
            EvalCheckStatus.SKIPPED,
            "LLM-as-judge is not configured for this deterministic offline case.",
            component="llm_judge",
        )
    return _result(
        case,
        EvalCheckKind.LLM_JUDGE,
        EvalCheckStatus.SKIPPED,
        "LLM-as-judge prompt is declared but execution is intentionally not part of local evals.",
        component="llm_judge",
        details={"prompt_id": case.llm_judge_prompt_id},
    )


def _result(
    case: EvalCase,
    kind: EvalCheckKind,
    status: EvalCheckStatus,
    message: str,
    *,
    component: str | None = None,
    source: str | None = None,
    details: Mapping[str, object] | None = None,
) -> EvalCheckResult:
    return EvalCheckResult(
        case_id=case.id,
        kind=kind,
        status=status,
        message=message,
        component=component,
        source=source,
        details=details or {},
    )


def _path_has_value(payload: Mapping[str, object], path: str) -> bool:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    if current is None:
        return False
    if isinstance(current, str):
        return current.strip() != ""
    if isinstance(current, list | tuple | dict):
        return len(current) > 0
    return True


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _collect_values_for_keys(value: object, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys:
                if isinstance(item, list | tuple | set):
                    found.update(str(entry) for entry in item if str(entry).strip())
                elif item is not None and str(item).strip():
                    found.add(str(item))
            found.update(_collect_values_for_keys(item, keys))
    elif isinstance(value, list | tuple):
        for item in value:
            found.update(_collect_values_for_keys(item, keys))
    return found


def _collect_datetime_values(value: object, keys: set[str]) -> tuple[datetime, ...]:
    found: list[datetime] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys:
                parsed = _datetime_from_value(item)
                if parsed is not None:
                    found.append(parsed)
            found.extend(_collect_datetime_values(item, keys))
    elif isinstance(value, list | tuple):
        for item in value:
            found.extend(_collect_datetime_values(item, keys))
    return tuple(found)


def _datetime_from_value(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime.combine(parsed_date, time.min, tzinfo=UTC)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _first_source_url(citations: tuple[Mapping[str, object], ...]) -> str | None:
    for citation in citations:
        source_url = citation.get("source_url")
        if isinstance(source_url, str) and source_url.strip():
            return source_url
    return None


def _text_blob(value: object) -> str:
    parts: list[str] = []
    _collect_text(value, parts)
    return "\n".join(parts)


def _collect_text(value: object, parts: list[str]) -> None:
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_text(item, parts)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _collect_text(item, parts)
