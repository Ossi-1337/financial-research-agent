from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from financial_research_agent.reports import (
    Citation,
    CitedResearchRun,
    CitedResearchRunStatus,
    CitedResearchRunStore,
    EvidenceSnippet,
    build_rag_messages,
    citation_artifacts_from_retrieval,
    ensure_citation_marker,
    missing_evidence_limitation,
)
from financial_research_agent.retrieval import (
    RetrievalChunk,
    RetrievalMatch,
    RetrievalQuery,
    RetrievalResult,
    RetrievalSourceKind,
)


def test_citation_contracts_are_immutable_and_format_marker() -> None:
    citation = Citation(
        id="C1",
        evidence_id="evidence:1",
        source_url="https://example.invalid/report.htm",
        retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
        quote_start=10,
        quote_end=20,
        quote="short quote",
    )

    assert citation.marker == "[C1]"
    assert citation.to_dict()["marker"] == "[C1]"
    with pytest.raises(FrozenInstanceError):
        citation.id = "C2"
    with pytest.raises(ValueError, match="quote_end"):
        Citation(
            id="C1",
            evidence_id="evidence:1",
            source_url="https://example.invalid/report.htm",
            retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
            quote_start=20,
            quote_end=10,
        )


def test_retrieval_result_converts_to_citations_with_source_metadata() -> None:
    result = RetrievalResult(
        query=RetrievalQuery(query="revenue"),
        matches=(
            RetrievalMatch(
                chunk=RetrievalChunk(
                    id="retrieval:chunk-1",
                    text="TEST TOOL OUTPUT revenue discussion with source metadata.",
                    source_kind=RetrievalSourceKind.FILING_CHUNK,
                    source_id="filing-chunk-1",
                    source_url="https://example.invalid/aapl-10k.htm",
                    document_id="filing-1",
                    section_heading="Item 7. Management Discussion",
                    metadata={
                        "cik": "320193",
                        "accession_number": "0000320193-25-000001",
                        "char_start": "100",
                    },
                ),
                score=0.91,
                rank=1,
            ),
        ),
        provider="local-vector",
        index_id="local-vector",
        generated_at=datetime(2026, 7, 5, tzinfo=UTC),
    )

    citations, evidence = citation_artifacts_from_retrieval(result)

    assert citations[0].id == "C1"
    assert citations[0].marker == "[C1]"
    assert citations[0].source_id == "filing-chunk-1"
    assert citations[0].chunk_id == "retrieval:chunk-1"
    assert citations[0].quote_start == 100
    assert citations[0].source_url == "https://example.invalid/aapl-10k.htm"
    assert evidence[0].citation_id == "C1"
    assert evidence[0].score == 0.91
    assert evidence[0].metadata["accession_number"] == "0000320193-25-000001"


def test_rag_prompt_requires_evidence_and_passes_only_snippets() -> None:
    evidence = (
        EvidenceSnippet(
            id="evidence:1",
            citation_id="C1",
            text="TEST TOOL OUTPUT evidence text",
            source_url="https://example.invalid/source.htm",
            retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
            score=0.88,
            section="Item 1. Business",
        ),
    )

    messages = build_rag_messages("What changed?", evidence)

    assert "using only the evidence snippets" in messages[0].content
    assert "Do not provide buy, sell, or hold recommendations" in messages[0].content
    assert "[C1]" in messages[1].content
    assert "TEST TOOL OUTPUT evidence text" in messages[1].content
    with pytest.raises(ValueError, match="evidence is required"):
        build_rag_messages("What changed?", ())


def test_answer_marker_guardrail_and_missing_evidence_limitation() -> None:
    citation = Citation(
        id="C1",
        evidence_id="evidence:1",
        source_url="https://example.invalid/source.htm",
        retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert ensure_citation_marker("Answer body", (citation,)).endswith("Sources: [C1]")
    assert ensure_citation_marker("Answer body [C1]", (citation,)) == "Answer body [C1]"
    assert "could not find stored evidence" in missing_evidence_limitation("Explain revenue")


def test_cited_research_run_store_persists_citation_data(tmp_path: Path) -> None:
    path = tmp_path / "report_runs.json"
    run = CitedResearchRun(
        id="research_run_1",
        query="Explain revenue",
        answer="Revenue grew [C1]",
        status=CitedResearchRunStatus.ANSWERED,
        created_at=datetime(2026, 7, 5, tzinfo=UTC),
        citations=(
            Citation(
                id="C1",
                evidence_id="evidence:1",
                source_url="https://example.invalid/source.htm",
                retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
            ),
        ),
        evidence=(
            EvidenceSnippet(
                id="evidence:1",
                citation_id="C1",
                text="TEST TOOL OUTPUT evidence",
                source_url="https://example.invalid/source.htm",
                retrieved_at=datetime(2026, 7, 5, tzinfo=UTC),
                score=0.9,
            ),
        ),
        provider="offline-test",
        model="offline-test",
    )

    CitedResearchRunStore(storage_path=path).save(run)
    reloaded = CitedResearchRunStore(storage_path=path)

    loaded = reloaded.get("research_run_1")
    assert loaded is not None
    assert loaded.citations[0].marker == "[C1]"
    assert loaded.evidence[0].text == "TEST TOOL OUTPUT evidence"
    assert reloaded.count() == 1
