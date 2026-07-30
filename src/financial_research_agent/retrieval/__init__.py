"""Local vector retrieval and RAG foundation contracts."""

from financial_research_agent.retrieval.contracts import (
    ChatRetrievalMetadata,
    IndexedChunk,
    RetrievalChunk,
    RetrievalError,
    RetrievalErrorCode,
    RetrievalIndexBuildResult,
    RetrievalIndexMetadata,
    RetrievalMatch,
    RetrievalProviderName,
    RetrievalQuery,
    RetrievalResult,
    RetrievalScoreKind,
    RetrievalSourceKind,
)
from financial_research_agent.retrieval.local_index import (
    LOCAL_VECTOR_INDEX_VERSION,
    LocalVectorIndex,
)
from financial_research_agent.retrieval.pipeline import (
    index_filing_result,
    retrieval_chunk_from_filing_chunk,
    search_index,
)

__all__ = [
    "LOCAL_VECTOR_INDEX_VERSION",
    "ChatRetrievalMetadata",
    "IndexedChunk",
    "LocalVectorIndex",
    "RetrievalChunk",
    "RetrievalError",
    "RetrievalErrorCode",
    "RetrievalIndexBuildResult",
    "RetrievalIndexMetadata",
    "RetrievalMatch",
    "RetrievalProviderName",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalScoreKind",
    "RetrievalSourceKind",
    "index_filing_result",
    "retrieval_chunk_from_filing_chunk",
    "search_index",
]
