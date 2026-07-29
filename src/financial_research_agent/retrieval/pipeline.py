from __future__ import annotations

from datetime import UTC, datetime

from financial_research_agent.filings import FilingChunk, FilingIngestionResult
from financial_research_agent.llm import EmbeddingProvider, EmbeddingRequest, ProviderError
from financial_research_agent.retrieval.contracts import (
    IndexedChunk,
    RetrievalChunk,
    RetrievalError,
    RetrievalErrorCode,
    RetrievalIndexBuildResult,
    RetrievalQuery,
    RetrievalResult,
    RetrievalSourceKind,
)
from financial_research_agent.retrieval.local_index import LocalVectorIndex


async def index_filing_result(
    result: FilingIngestionResult,
    *,
    index: LocalVectorIndex,
    embedding_provider: EmbeddingProvider,
    embedding_model: str | None = None,
    replace_company: bool = True,
    now: datetime | None = None,
) -> RetrievalIndexBuildResult:
    indexed_at = now or datetime.now(UTC)
    chunks = tuple(
        retrieval_chunk_from_filing_chunk(
            chunk,
            cik=result.company.cik,
            company_id=result.company.company_id,
            legal_name=result.company.legal_name,
        )
        for chunk in result.chunks
    )
    if not chunks:
        return RetrievalIndexBuildResult(
            provider=index.provider,
            index_id=index.index_id,
            indexed_count=0,
            skipped_count=0,
            embedding_provider=None,
            embedding_model=None,
            indexed_at=indexed_at,
            warnings=("No filing chunks were available to index.", *result.warnings),
        )

    try:
        response = await embedding_provider.embed(
            EmbeddingRequest(
                input_texts=tuple(chunk.text for chunk in chunks),
                model=embedding_model,
                metadata={"purpose": "retrieval_index"},
            )
        )
    except ProviderError as exc:
        raise RetrievalError(
            code=RetrievalErrorCode.EMBEDDING_FAILED,
            message=exc.message,
            provider=index.provider,
            retryable=exc.retryable,
        ) from exc

    if len(response.embeddings) != len(chunks):
        raise RetrievalError(
            code=RetrievalErrorCode.EMBEDDING_FAILED,
            message="Embedding provider returned a different number of vectors than chunks.",
            provider=index.provider,
        )

    records = tuple(
        IndexedChunk(
            chunk=chunk,
            embedding=embedding,
            embedding_provider=response.provider,
            embedding_model=response.model,
            indexed_at=indexed_at,
        )
        for chunk, embedding in zip(chunks, response.embeddings, strict=True)
    )
    if replace_company:
        index.delete_by_metadata("cik", result.company.cik)
    index.upsert(records)
    return RetrievalIndexBuildResult(
        provider=index.provider,
        index_id=index.index_id,
        indexed_count=len(records),
        skipped_count=0,
        embedding_provider=response.provider,
        embedding_model=response.model,
        indexed_at=indexed_at,
        warnings=result.warnings,
    )


async def search_index(
    query: RetrievalQuery,
    *,
    index: LocalVectorIndex,
    embedding_provider: EmbeddingProvider,
    embedding_model: str | None = None,
    now: datetime | None = None,
) -> RetrievalResult:
    try:
        response = await embedding_provider.embed(
            EmbeddingRequest(
                input_texts=(query.query,),
                model=embedding_model,
                metadata={"purpose": "retrieval_query"},
            )
        )
    except ProviderError as exc:
        raise RetrievalError(
            code=RetrievalErrorCode.EMBEDDING_FAILED,
            message=exc.message,
            provider=index.provider,
            retryable=exc.retryable,
        ) from exc

    if len(response.embeddings) != 1:
        raise RetrievalError(
            code=RetrievalErrorCode.EMBEDDING_FAILED,
            message="Embedding provider returned an unexpected number of query vectors.",
            provider=index.provider,
        )
    return index.search(query, query_embedding=response.embeddings[0], now=now)


def retrieval_chunk_from_filing_chunk(
    chunk: FilingChunk,
    *,
    cik: str,
    company_id: str | None = None,
    legal_name: str | None = None,
) -> RetrievalChunk:
    metadata = {
        **dict(chunk.metadata),
        "cik": cik,
        "filing_id": chunk.filing_id,
        "chunk_index": str(chunk.chunk_index),
        "accession_number": chunk.accession_number,
        "form_type": chunk.form_type,
        "char_start": str(chunk.char_start),
        "char_end": str(chunk.char_end),
    }
    if company_id is not None:
        metadata["company_id"] = company_id
    if legal_name is not None:
        metadata["legal_name"] = legal_name
    if chunk.source_region is not None:
        metadata["page_number"] = str(chunk.source_region.page_number)
        metadata["region_left"] = str(chunk.source_region.left)
        metadata["region_top"] = str(chunk.source_region.top)
        metadata["region_right"] = str(chunk.source_region.right)
        metadata["region_bottom"] = str(chunk.source_region.bottom)
    if chunk.extraction_method is not None:
        metadata["extraction_method"] = chunk.extraction_method.value
    return RetrievalChunk(
        id=f"retrieval:{chunk.id}",
        text=chunk.text,
        source_kind=RetrievalSourceKind.FILING_CHUNK,
        source_id=chunk.id,
        source_url=chunk.source_url,
        document_id=chunk.filing_id,
        title=f"{chunk.form_type} filing chunk",
        section_heading=chunk.section_heading,
        metadata=metadata,
    )
