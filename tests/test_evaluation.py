from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from financial_research_agent.evaluation import (
    EvalArtifact,
    EvalArtifactKind,
    EvalCase,
    EvalCheckKind,
    EvalCheckStatus,
    EvalDatasetLabel,
    EvalSuiteStatus,
    default_eval_artifacts,
    default_eval_cases,
    evaluate_case,
    run_default_offline_evaluations,
)

NOW = datetime(2026, 7, 6, 12, tzinfo=UTC)


def test_eval_case_contract_is_immutable_and_validates_inputs() -> None:
    case = EvalCase(
        id="fixture:test",
        query="What changed?",
        artifact_kind=EvalArtifactKind.CITED_ANSWER,
        dataset_label=EvalDatasetLabel.FIXTURE,
        description="Fixture case.",
        required_schema_paths=("answer",),
    )

    assert case.to_dict()["dataset_label"] == "fixture"
    with pytest.raises(FrozenInstanceError):
        case.query = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="min_citations"):
        EvalCase(
            id="bad",
            query="Bad",
            artifact_kind=EvalArtifactKind.CITED_ANSWER,
            dataset_label=EvalDatasetLabel.FIXTURE,
            description="Bad",
            min_citations=-1,
        )


def test_default_eval_dataset_is_offline_fixture_labeled() -> None:
    cases = default_eval_cases()
    artifacts = default_eval_artifacts(now=NOW)

    assert len(cases) == 3
    assert {case.dataset_label for case in cases} == {EvalDatasetLabel.FIXTURE}
    assert all(case.metadata["fixture"] == "true" for case in cases)
    assert {artifact.case_id for artifact in artifacts} == {case.id for case in cases}
    assert all(artifact.provider == "fixture" for artifact in artifacts)


def test_default_offline_evaluation_suite_passes_deterministically() -> None:
    result = run_default_offline_evaluations(now=NOW)
    payload = result.to_dict()

    assert result.status == EvalSuiteStatus.PASSED
    assert payload["id"] == "default-offline-fixture"
    assert payload["case_count"] == 3
    assert payload["failed_count"] == 0
    assert any(
        check["kind"] == "llm_judge" and check["status"] == "skipped"
        for case_result in payload["case_results"]
        for check in case_result["checks"]
    )


def test_schema_and_citation_failures_identify_component_and_source() -> None:
    case = EvalCase(
        id="fixture:missing-citation",
        query="Explain revenue.",
        artifact_kind=EvalArtifactKind.CITED_ANSWER,
        dataset_label=EvalDatasetLabel.FIXTURE,
        description="Fixture missing citation.",
        required_schema_paths=("answer", "citations"),
        min_citations=1,
        required_evidence_ids=("fixture:evidence:missing",),
    )
    artifact = EvalArtifact(
        case_id=case.id,
        artifact_kind=EvalArtifactKind.CITED_ANSWER,
        payload={"answer": "Unsupported answer.", "citations": []},
    )

    result = evaluate_case(case=case, artifact=artifact, now=NOW)
    citation_check = next(
        check for check in result.checks if check.kind == EvalCheckKind.CITATION_COVERAGE
    )

    assert result.status == EvalSuiteStatus.FAILED
    assert citation_check.status == EvalCheckStatus.FAILED
    assert citation_check.component == "retrieval"
    assert citation_check.details["missing_evidence_ids"] == ["fixture:evidence:missing"]


def test_source_freshness_and_hallucination_guardrails_fail() -> None:
    case = EvalCase(
        id="fixture:stale-and-claim",
        query="Explain revenue.",
        artifact_kind=EvalArtifactKind.CITED_ANSWER,
        dataset_label=EvalDatasetLabel.FIXTURE,
        description="Fixture stale source and forbidden claim.",
        required_schema_paths=("answer", "citations"),
        min_citations=1,
        max_source_age_days=7,
        forbidden_claim_patterns=(r"\bguaranteed\b",),
    )
    artifact = EvalArtifact(
        case_id=case.id,
        artifact_kind=EvalArtifactKind.CITED_ANSWER,
        payload={
            "answer": "Revenue is guaranteed to improve [C1].",
            "citations": [
                {
                    "id": "C1",
                    "source_url": "https://example.invalid/stale.htm",
                    "retrieved_at": (NOW - timedelta(days=30)).isoformat(),
                }
            ],
        },
    )

    result = evaluate_case(case=case, artifact=artifact, now=NOW)
    statuses = {check.kind: check.status for check in result.checks}

    assert statuses[EvalCheckKind.SOURCE_FRESHNESS] == EvalCheckStatus.FAILED
    assert statuses[EvalCheckKind.HALLUCINATION_GUARDRAIL] == EvalCheckStatus.FAILED


def test_refusal_and_traceability_checks() -> None:
    case = EvalCase(
        id="fixture:trace-refusal",
        query="No evidence?",
        artifact_kind=EvalArtifactKind.SYNTHESIS_REPORT,
        dataset_label=EvalDatasetLabel.FIXTURE,
        description="Fixture trace and refusal.",
        required_schema_paths=("summary",),
        expects_refusal=True,
        required_refusal_terms=("cannot verify",),
        required_trace_components=("synthesis",),
    )
    artifact = EvalArtifact(
        case_id=case.id,
        artifact_kind=EvalArtifactKind.SYNTHESIS_REPORT,
        payload={"summary": "I cannot verify this without evidence."},
        trace={"events": [{"component": "synthesis"}]},
    )

    result = evaluate_case(case=case, artifact=artifact, now=NOW)
    statuses = {check.kind: check.status for check in result.checks}

    assert statuses[EvalCheckKind.REFUSAL_BEHAVIOR] == EvalCheckStatus.PASSED
    assert statuses[EvalCheckKind.TRACEABILITY] == EvalCheckStatus.PASSED
