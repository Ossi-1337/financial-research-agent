"""Filing and document ingestion contracts, SEC adapter, extraction, and storage."""

from financial_research_agent.filings.contracts import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingIngestionResult,
    FilingProvider,
    FilingProviderName,
    FilingSource,
)
from financial_research_agent.filings.extraction import (
    build_chunks,
    build_document_chunks,
    detect_document_format,
    extract_document_text,
)
from financial_research_agent.filings.sec_edgar import SECEDGARFilingProvider
from financial_research_agent.filings.store import FilingStore
from financial_research_agent.settings import Settings


def create_default_filing_provider(settings: Settings) -> FilingProvider:
    provider = settings.data_sources.filing_provider
    if provider != FilingProviderName.SEC_EDGAR.value:
        raise ValueError(f"Unsupported filing provider: {provider}")
    filings_root = settings.local_paths.data_dir / "filings"
    return SECEDGARFilingProvider(
        raw_documents_dir=filings_root / "raw",
        extracted_text_dir=filings_root / "text",
        max_document_bytes=settings.data_sources.filing_max_document_bytes,
        pdf_max_document_bytes=settings.data_sources.pdf_max_document_bytes,
        pdf_max_pages=settings.data_sources.pdf_max_pages,
        pdf_max_extracted_chars=settings.data_sources.pdf_max_extracted_chars,
        pdf_extraction_timeout_seconds=settings.data_sources.pdf_extraction_timeout_seconds,
        user_agent=settings.data_sources.sec_user_agent,
    )


__all__ = [
    "FilingChunk",
    "FilingCompany",
    "FilingDocument",
    "FilingDocumentFormat",
    "FilingError",
    "FilingErrorCode",
    "FilingIngestionResult",
    "FilingProvider",
    "FilingProviderName",
    "FilingSource",
    "FilingStore",
    "SECEDGARFilingProvider",
    "build_chunks",
    "build_document_chunks",
    "create_default_filing_provider",
    "detect_document_format",
    "extract_document_text",
]
