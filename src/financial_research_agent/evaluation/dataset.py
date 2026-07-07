from __future__ import annotations

from datetime import UTC, datetime, timedelta

from financial_research_agent.evaluation.contracts import (
    EvalArtifact,
    EvalArtifactKind,
    EvalCase,
    EvalDatasetLabel,
)

DEFAULT_EVALUATION_SUITE_ID = "default-offline-fixture"
DEFAULT_FORBIDDEN_CLAIM_PATTERNS = (
    r"\bguaranteed\b",
    r"\byou should (buy|sell|hold)\b",
    r"\bprice target\b",
    r"\bwill definitely\b",
)


def default_eval_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            id="fixture:cited-answer:revenue-citation",
            query="What does the stored filing say about revenue?",
            artifact_kind=EvalArtifactKind.CITED_ANSWER,
            dataset_label=EvalDatasetLabel.FIXTURE,
            description=(
                "Fixture cited-answer case that checks schema, citation marker, evidence id, "
                "source freshness, and unsupported-claim guardrails."
            ),
            required_schema_paths=("answer", "citations", "evidence"),
            min_citations=1,
            required_citation_markers=("[C1]",),
            required_evidence_ids=("fixture:evidence:revenue",),
            max_source_age_days=14,
            forbidden_claim_patterns=DEFAULT_FORBIDDEN_CLAIM_PATTERNS,
            metadata={"fixture": "true", "source": "TEST TOOL OUTPUT"},
        ),
        EvalCase(
            id="fixture:cited-answer:no-evidence-refusal",
            query="What are the current margins without stored evidence?",
            artifact_kind=EvalArtifactKind.CITED_ANSWER,
            dataset_label=EvalDatasetLabel.FIXTURE,
            description=(
                "Fixture refusal case that checks missing-evidence behavior without adding "
                "unsupported company facts."
            ),
            required_schema_paths=("answer", "limitations"),
            expects_refusal=True,
            required_refusal_terms=("could not find stored evidence", "cannot verify"),
            forbidden_claim_patterns=DEFAULT_FORBIDDEN_CLAIM_PATTERNS,
            metadata={"fixture": "true", "source": "TEST TOOL OUTPUT"},
        ),
        EvalCase(
            id="fixture:synthesis:traceable-scenarios",
            query="Summarize the current situation and scenarios.",
            artifact_kind=EvalArtifactKind.SYNTHESIS_REPORT,
            dataset_label=EvalDatasetLabel.FIXTURE,
            description=(
                "Fixture synthesis case that checks report schema, evidence ids, scenario "
                "traceability, and no recommendation leakage."
            ),
            required_schema_paths=(
                "summary",
                "sections.current_situation",
                "scenarios.upside",
                "scenarios.downside",
                "evidence_ids",
                "no_recommendation_notice",
            ),
            required_evidence_ids=("fixture:evidence:revenue",),
            required_trace_components=("financial_report_analysis", "synthesis"),
            forbidden_claim_patterns=DEFAULT_FORBIDDEN_CLAIM_PATTERNS,
            metadata={"fixture": "true", "source": "TEST TOOL OUTPUT"},
        ),
    )


def default_eval_artifacts(
    *,
    now: datetime | None = None,
) -> tuple[EvalArtifact, ...]:
    generated_at = now or datetime.now(UTC)
    retrieved_at = generated_at - timedelta(days=1)
    retrieved_text = retrieved_at.isoformat()
    return (
        EvalArtifact(
            case_id="fixture:cited-answer:revenue-citation",
            artifact_kind=EvalArtifactKind.CITED_ANSWER,
            payload={
                "answer": (
                    "TEST TOOL OUTPUT filing evidence says revenue increased in the stored "
                    "fixture. Sources: [C1]"
                ),
                "citations": [
                    {
                        "id": "C1",
                        "marker": "[C1]",
                        "evidence_id": "fixture:evidence:revenue",
                        "source_url": "https://example.invalid/fixture-10k.htm",
                        "retrieved_at": retrieved_text,
                    }
                ],
                "evidence": [
                    {
                        "id": "fixture:evidence:revenue",
                        "citation_id": "C1",
                        "text": "TEST TOOL OUTPUT revenue evidence.",
                        "source_url": "https://example.invalid/fixture-10k.htm",
                        "retrieved_at": retrieved_text,
                    }
                ],
            },
            provider="fixture",
            model="deterministic-check",
        ),
        EvalArtifact(
            case_id="fixture:cited-answer:no-evidence-refusal",
            artifact_kind=EvalArtifactKind.CITED_ANSWER,
            payload={
                "answer": (
                    "I could not find stored evidence for the requested current margin "
                    "figures, so I cannot verify that claim."
                ),
                "citations": [],
                "limitations": ["Missing stored evidence for current margins."],
            },
            provider="fixture",
            model="deterministic-check",
        ),
        EvalArtifact(
            case_id="fixture:synthesis:traceable-scenarios",
            artifact_kind=EvalArtifactKind.SYNTHESIS_REPORT,
            payload={
                "summary": "Fixture synthesis from stored specialist handoffs.",
                "sections": {
                    "current_situation": [
                        {
                            "title": "Revenue evidence",
                            "summary": "Stored fixture evidence supports revenue discussion.",
                            "evidence_ids": ["fixture:evidence:revenue"],
                            "source_handoff_ids": ["handoff_report"],
                        }
                    ]
                },
                "scenarios": {
                    "upside": {
                        "condition": "If revenue evidence remains supported by filings.",
                        "potential_development": (
                            "When evidence coverage improves, confidence can increase."
                        ),
                        "evidence_ids": ["fixture:evidence:revenue"],
                    },
                    "downside": {
                        "condition": "If evidence remains incomplete.",
                        "potential_development": "When gaps persist, uncertainty remains high.",
                        "limitations": ["Fixture source coverage is intentionally limited."],
                    },
                },
                "evidence_ids": ["fixture:evidence:revenue"],
                "no_recommendation_notice": (
                    "This synthesis does not provide buy, sell, hold, price-target, or "
                    "personalized investment advice."
                ),
            },
            trace={
                "events": [
                    {"component": "financial_report_analysis", "status": "succeeded"},
                    {"component": "synthesis", "status": "succeeded"},
                ]
            },
            provider="fixture",
            model="deterministic-check",
        ),
    )
