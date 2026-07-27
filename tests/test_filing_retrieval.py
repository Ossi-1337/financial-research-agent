from __future__ import annotations

from datetime import UTC, date, datetime

from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingIngestionResult,
    FilingSource,
)
from financial_research_agent.report_analysis import (
    FilingRetrievalMethod,
    retrieve_filing_chunks,
)


def test_filing_retrieval_uses_stable_lexical_ranking() -> None:
    result = _filing_result()

    matches = retrieve_filing_chunks(result, ("risk", "liquidity"), limit=2)

    assert [match.chunk.id for match in matches] == ["chunk:risk", "chunk:liquidity"]
    assert all(match.method == FilingRetrievalMethod.LEXICAL for match in matches)
    assert matches[0].score > matches[1].score


def test_filing_retrieval_can_apply_optional_vector_reranking() -> None:
    result = _filing_result()

    matches = retrieve_filing_chunks(
        result,
        ("risk", "liquidity"),
        limit=2,
        vector_reranker=lambda chunks, query: {
            chunk.id: 1.0 if chunk.id == "chunk:liquidity" else 0.1 for chunk in chunks
        },
    )

    assert [match.chunk.id for match in matches] == ["chunk:liquidity", "chunk:risk"]
    assert all(match.method == FilingRetrievalMethod.VECTOR_RERANKED for match in matches)


def test_filing_retrieval_returns_no_unmatched_or_missing_chunks() -> None:
    assert retrieve_filing_chunks(None, ("risk",), limit=3) == ()
    assert retrieve_filing_chunks(_filing_result(), ("not-present",), limit=3) == ()


def _filing_result() -> FilingIngestionResult:
    company = FilingCompany(
        cik="0000000001",
        company_id="fixture:company:test",
        legal_name="TEST TOOL OUTPUT COMPANY",
    )
    source = FilingSource(
        provider="sec-edgar",
        provider_status="test fixture",
        source_url="https://example.invalid/submissions.json",
        retrieved_at=datetime(2026, 7, 27, tzinfo=UTC),
        data_as_of=date(2026, 7, 26),
        attribution="test fixture",
    )
    chunks = (
        FilingChunk(
            id="chunk:risk",
            filing_id="filing:test",
            chunk_index=0,
            text="risk risk risk liquidity",
            char_start=0,
            char_end=24,
            source_url="https://example.invalid/filing",
            accession_number="0000000001-26-000001",
            form_type="10-K",
        ),
        FilingChunk(
            id="chunk:liquidity",
            filing_id="filing:test",
            chunk_index=1,
            text="liquidity position remained stable",
            char_start=25,
            char_end=60,
            source_url="https://example.invalid/filing",
            accession_number="0000000001-26-000001",
            form_type="10-K",
        ),
        FilingChunk(
            id="chunk:other",
            filing_id="filing:test",
            chunk_index=2,
            text="unrelated TEST TOOL OUTPUT",
            char_start=61,
            char_end=87,
            source_url="https://example.invalid/filing",
            accession_number="0000000001-26-000001",
            form_type="10-K",
        ),
    )
    return FilingIngestionResult(
        company=company,
        filings=(),
        chunks=chunks,
        source=source,
        warnings=("test fixture",),
    )
