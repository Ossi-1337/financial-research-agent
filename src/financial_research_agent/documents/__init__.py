"""Provider-neutral normalized document extraction."""

from financial_research_agent.documents.contracts import (
    DocumentExtractionError,
    DocumentExtractionErrorCode,
    DocumentExtractionMethod,
    DocumentExtractionResult,
    DocumentExtractionStatus,
    DocumentExtractor,
    DocumentFormat,
    DocumentPage,
    DocumentRegion,
    NormalizedDocument,
)
from financial_research_agent.documents.pdf import PDFDocumentExtractor

__all__ = [
    "DocumentExtractionError",
    "DocumentExtractionErrorCode",
    "DocumentExtractionMethod",
    "DocumentExtractionResult",
    "DocumentExtractionStatus",
    "DocumentExtractor",
    "DocumentFormat",
    "DocumentPage",
    "DocumentRegion",
    "NormalizedDocument",
    "PDFDocumentExtractor",
]
