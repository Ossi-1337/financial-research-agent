from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingIngestionResult,
    FilingSource,
)
from financial_research_agent.llm import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelMetadata,
    ProviderCapability,
    TokenUsage,
)
from financial_research_agent.retrieval import (
    ChatRetrievalMetadata,
    IndexedChunk,
    LocalVectorIndex,
    RetrievalChunk,
    RetrievalError,
    RetrievalErrorCode,
    RetrievalQuery,
    RetrievalSourceKind,
    index_filing_result,
    retrieval_chunk_from_filing_chunk,
    search_index,
)


def test_chat_retrieval_metadata_is_version_safe() -> None:
    metadata = ChatRetrievalMetadata(
        query="latest filing risk evidence",
        specialist_roles=("financial-report", "synthesis"),
        methods=("lexical",),
        evidence_ids=("filing:chunk:1",),
        duration_ms=9,
    )

    restored = ChatRetrievalMetadata.from_dict(metadata.to_dict())

    assert restored == metadata
    with pytest.raises(FrozenInstanceError):
        metadata.query = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="duration_ms"):
        ChatRetrievalMetadata(duration_ms=-1)


def test_retrieval_contracts_are_immutable_and_validate_inputs() -> None:
    chunk = RetrievalChunk(
        id="chunk-1",
        text="TEST TOOL OUTPUT revenue growth discussion",
        source_kind=RetrievalSourceKind.FILING_CHUNK,
        source_id="filing-chunk-1",
        source_url="https://example.invalid/filing.htm",
        metadata={"cik": "320193"},
    )
    indexed = IndexedChunk(
        chunk=chunk,
        embedding=(1.0, 0.0, 0.0),
        embedding_provider="keyword-fixture",
        embedding_model="keyword-fixture",
        indexed_at=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert indexed.id == "chunk-1"
    assert chunk.snippet(max_chars=20).endswith("...")
    with pytest.raises(FrozenInstanceError):
        chunk.id = "mutated"
    with pytest.raises(TypeError):
        chunk.metadata["cik"] = "1"
    with pytest.raises(ValueError, match="top_k"):
        RetrievalQuery(query="revenue", top_k=0)


def test_local_vector_index_search_returns_source_linked_chunk(tmp_path: Path) -> None:
    index = LocalVectorIndex(storage_path=tmp_path / "vector_index.json")
    chunk = _retrieval_chunk(
        "chunk-revenue",
        "TEST TOOL OUTPUT revenue growth and cash flow",
        {"cik": "320193", "form_type": "10-K"},
    )
    other = _retrieval_chunk(
        "chunk-risk",
        "TEST TOOL OUTPUT risk factors and litigation",
        {"cik": "320193", "form_type": "10-K"},
    )
    index.upsert(
        (
            _indexed_chunk(chunk, (1.0, 1.0, 0.0)),
            _indexed_chunk(other, (0.0, 0.0, 1.0)),
        )
    )

    result = index.search(
        RetrievalQuery(query="revenue", top_k=1, filters={"cik": "320193"}),
        query_embedding=(1.0, 0.0, 0.0),
        now=datetime(2026, 7, 5, tzinfo=UTC),
    )

    assert result.matches[0].chunk.id == "chunk-revenue"
    assert result.matches[0].chunk.source_url == "https://example.invalid/chunk-revenue.htm"
    assert result.matches[0].score > 0.7
    assert result.to_dict()["matches"][0]["snippet"].startswith("TEST TOOL OUTPUT revenue")


def test_local_vector_index_persists_and_clears_records(tmp_path: Path) -> None:
    path = tmp_path / "vector_index.json"
    index = LocalVectorIndex(storage_path=path)
    index.upsert((_indexed_chunk(_retrieval_chunk("chunk-1", "cash flow", {}), (0.0, 1.0, 0.0)),))

    reloaded = LocalVectorIndex(storage_path=path)
    assert reloaded.metadata().record_count == 1

    deleted = reloaded.clear()

    assert deleted == 1
    assert not path.exists()


def test_local_vector_index_rejects_dimension_mismatch() -> None:
    index = LocalVectorIndex()
    index.upsert((_indexed_chunk(_retrieval_chunk("chunk-1", "cash flow", {}), (1.0, 0.0)),))

    with pytest.raises(RetrievalError) as exc:
        index.upsert((_indexed_chunk(_retrieval_chunk("chunk-2", "risk factors", {}), (1.0,)),))

    assert exc.value.code == RetrievalErrorCode.VECTOR_DIMENSION_MISMATCH


def test_local_vector_index_rejects_naive_updated_at(tmp_path: Path) -> None:
    path = tmp_path / "vector_index.json"
    path.write_text(
        '{"version":1,"updated_at":"2026-07-05T00:00:00","records":[]}',
        encoding="utf-8",
    )

    with pytest.raises(RetrievalError) as exc:
        LocalVectorIndex(storage_path=path)

    assert exc.value.code == RetrievalErrorCode.MALFORMED_INDEX


def test_index_filing_result_uses_swappable_embedding_provider() -> None:
    result = _filing_result()
    index = LocalVectorIndex()
    provider = KeywordEmbeddingProvider(provider="keyword-a", model="keyword-model-a")

    build = asyncio.run(
        index_filing_result(
            result,
            index=index,
            embedding_provider=provider,
            now=datetime(2026, 7, 5, tzinfo=UTC),
        )
    )
    search = asyncio.run(
        search_index(
            RetrievalQuery(query="revenue growth", top_k=1),
            index=index,
            embedding_provider=KeywordEmbeddingProvider(
                provider="keyword-b",
                model="keyword-model-b",
            ),
            now=datetime(2026, 7, 5, tzinfo=UTC),
        )
    )

    assert build.indexed_count == 2
    assert build.embedding_provider == "keyword-a"
    assert search.matches[0].chunk.metadata["accession_number"] == "0000320193-25-000001"
    assert search.matches[0].chunk.metadata["cik"] == "320193"
    assert search.matches[0].chunk.source_url.endswith("aapl-20251231.htm")


def test_index_filing_result_handles_empty_chunks_without_embedding_call() -> None:
    source = _filing_source()
    result = FilingIngestionResult(
        company=FilingCompany(cik="320193"),
        filings=(),
        chunks=(),
        source=source,
    )
    provider = KeywordEmbeddingProvider()

    build = asyncio.run(
        index_filing_result(result, index=LocalVectorIndex(), embedding_provider=provider)
    )

    assert build.indexed_count == 0
    assert build.embedding_provider is None
    assert "No filing chunks" in build.warnings[0]
    assert provider.calls == []


def test_search_empty_index_returns_structured_error() -> None:
    with pytest.raises(RetrievalError) as exc:
        LocalVectorIndex().search(
            RetrievalQuery(query="revenue"),
            query_embedding=(1.0, 0.0, 0.0),
        )

    assert exc.value.code == RetrievalErrorCode.INDEX_EMPTY
    assert exc.value.to_dict()["error"] == "retrieval_error"


def test_retrieval_chunk_from_filing_chunk_preserves_citation_metadata() -> None:
    filing_chunk = _filing_result().chunks[0]

    chunk = retrieval_chunk_from_filing_chunk(
        filing_chunk,
        cik="320193",
        company_id="fixture:company:apple",
        legal_name="TEST TOOL OUTPUT APPLE INC.",
    )

    assert chunk.id == f"retrieval:{filing_chunk.id}"
    assert chunk.source_kind == RetrievalSourceKind.FILING_CHUNK
    assert chunk.source_id == filing_chunk.id
    assert chunk.metadata["form_type"] == "10-K"
    assert chunk.metadata["company_id"] == "fixture:company:apple"
    assert chunk.section_heading == "Item 1. Business"


def _retrieval_chunk(id: str, text: str, metadata: dict[str, str]) -> RetrievalChunk:
    return RetrievalChunk(
        id=id,
        text=text,
        source_kind=RetrievalSourceKind.FILING_CHUNK,
        source_id=f"source:{id}",
        source_url=f"https://example.invalid/{id}.htm",
        metadata=metadata,
    )


def _indexed_chunk(chunk: RetrievalChunk, embedding: tuple[float, ...]) -> IndexedChunk:
    return IndexedChunk(
        chunk=chunk,
        embedding=embedding,
        embedding_provider="keyword-fixture",
        embedding_model="keyword-fixture",
        indexed_at=datetime(2026, 7, 5, tzinfo=UTC),
    )


def _filing_result() -> FilingIngestionResult:
    company = FilingCompany(
        cik="320193",
        company_id="fixture:company:apple",
        legal_name="TEST TOOL OUTPUT APPLE INC.",
    )
    source = _filing_source()
    filing = FilingDocument(
        id="fixture:filing:10-k",
        company=company,
        form_type="10-K",
        accession_number="0000320193-25-000001",
        filing_date=datetime(2026, 1, 31, tzinfo=UTC).date(),
        report_date=datetime(2025, 12, 31, tzinfo=UTC).date(),
        publication_date=datetime(2026, 1, 31, tzinfo=UTC).date(),
        document_url="https://example.invalid/aapl-20251231.htm",
        source_url="https://example.invalid/submissions/CIK0000320193.json",
        document_format=FilingDocumentFormat.HTML,
        retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
        local_raw_path="fixture/raw/aapl-20251231.htm",
        local_text_path="fixture/text/aapl-20251231.txt",
        source=source,
        chunk_ids=("fixture:filing:10-k:chunk:0", "fixture:filing:10-k:chunk:1"),
    )
    chunks = (
        FilingChunk(
            id="fixture:filing:10-k:chunk:0",
            filing_id=filing.id,
            chunk_index=0,
            text="TEST TOOL OUTPUT revenue growth and cash flow discussion.",
            char_start=0,
            char_end=57,
            section_heading="Item 1. Business",
            source_url=filing.document_url,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            metadata={"fixture": "true"},
        ),
        FilingChunk(
            id="fixture:filing:10-k:chunk:1",
            filing_id=filing.id,
            chunk_index=1,
            text="TEST TOOL OUTPUT risk factors and litigation discussion.",
            char_start=58,
            char_end=111,
            section_heading="Item 1A. Risk Factors",
            source_url=filing.document_url,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            metadata={"fixture": "true"},
        ),
    )
    return FilingIngestionResult(company=company, filings=(filing,), chunks=chunks, source=source)


def _filing_source() -> FilingSource:
    return FilingSource(
        provider="sec-edgar",
        provider_status="test fixture",
        source_url="https://example.invalid/submissions/CIK0000320193.json",
        retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
        data_as_of=datetime(2026, 1, 31, tzinfo=UTC).date(),
        attribution="test fixture",
    )


class KeywordEmbeddingProvider:
    def __init__(self, *, provider: str = "keyword-fixture", model: str = "keyword-model") -> None:
        self.provider = provider
        self.model = model
        self.calls: list[EmbeddingRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider=self.provider,
            model=self.model,
            capabilities=(ProviderCapability.EMBEDDINGS,),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.calls.append(request)
        return EmbeddingResponse(
            embeddings=tuple(_keyword_vector(text) for text in request.input_texts),
            provider=self.provider,
            model=request.model or self.model,
            usage=TokenUsage(input_tokens=sum(len(text.split()) for text in request.input_texts)),
        )


def _keyword_vector(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    return (
        float(lowered.count("revenue") + lowered.count("growth")),
        float(lowered.count("cash")),
        float(lowered.count("risk") + lowered.count("litigation")),
    )
