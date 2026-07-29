from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from financial_research_agent.documents import (
    DocumentExtractionMethod,
    DocumentExtractionStatus,
    DocumentRegion,
)


class FilingProviderName(StrEnum):
    SEC_EDGAR = "sec-edgar"


class FilingDocumentFormat(StrEnum):
    HTML = "html"
    TEXT = "text"
    PDF = "pdf"
    UNSUPPORTED = "unsupported"


class FilingErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    NOT_FOUND = "not_found"
    UNSUPPORTED_FORMAT = "unsupported_format"
    DOCUMENT_TOO_LARGE = "document_too_large"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    OCR_REQUIRED = "ocr_required"
    ENCRYPTED_DOCUMENT = "encrypted_document"
    EXTRACTION_TIMEOUT = "extraction_timeout"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True, slots=True)
class FilingCompany:
    cik: str
    company_id: str | None = None
    legal_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cik", _normalize_cik(self.cik))
        object.__setattr__(self, "company_id", _optional_text(self.company_id))
        object.__setattr__(self, "legal_name", _optional_text(self.legal_name))

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)

    def to_dict(self) -> dict[str, object]:
        return {
            "cik": self.cik,
            "padded_cik": self.padded_cik,
            "company_id": self.company_id,
            "legal_name": self.legal_name,
        }


@dataclass(frozen=True, slots=True)
class FilingSource:
    provider: str
    provider_status: str
    source_url: str
    retrieved_at: datetime
    attribution: str
    data_as_of: date | None = None
    freshness_warning: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(
            self,
            "provider_status",
            _require_text("provider_status", self.provider_status),
        )
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(self, "attribution", _require_text("attribution", self.attribution))
        object.__setattr__(self, "freshness_warning", _optional_text(self.freshness_warning))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_status": self.provider_status,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of is not None else None,
            "attribution": self.attribution,
            "freshness_warning": self.freshness_warning,
        }


@dataclass(frozen=True, slots=True)
class FilingDocument:
    id: str
    company: FilingCompany
    form_type: str
    accession_number: str
    filing_date: date
    report_date: date | None
    document_url: str
    source_url: str
    document_format: FilingDocumentFormat
    retrieved_at: datetime
    local_raw_path: str
    local_text_path: str
    source: FilingSource
    publication_date: date | None = None
    chunk_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    extraction_status: DocumentExtractionStatus | None = None
    extraction_method: DocumentExtractionMethod | None = None
    page_count: int | None = None
    missing_text_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.company, FilingCompany):
            raise ValueError("company must be a FilingCompany")
        object.__setattr__(self, "form_type", _require_text("form_type", self.form_type).upper())
        object.__setattr__(
            self,
            "accession_number",
            _require_text("accession_number", self.accession_number),
        )
        object.__setattr__(self, "document_url", _require_text("document_url", self.document_url))
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "document_format", FilingDocumentFormat(self.document_format))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(
            self,
            "local_raw_path",
            _require_text("local_raw_path", self.local_raw_path),
        )
        object.__setattr__(
            self,
            "local_text_path",
            _require_text("local_text_path", self.local_text_path),
        )
        if not isinstance(self.source, FilingSource):
            raise ValueError("source must be a FilingSource")
        object.__setattr__(self, "chunk_ids", _text_tuple("chunk_ids", self.chunk_ids))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        if self.extraction_status is not None:
            object.__setattr__(
                self,
                "extraction_status",
                DocumentExtractionStatus(self.extraction_status),
            )
        if self.extraction_method is not None:
            object.__setattr__(
                self,
                "extraction_method",
                DocumentExtractionMethod(self.extraction_method),
            )
        if self.page_count is not None and self.page_count <= 0:
            raise ValueError("page_count must be positive")
        missing_text_pages = tuple(self.missing_text_pages)
        if any(page <= 0 for page in missing_text_pages):
            raise ValueError("missing_text_pages must contain positive page numbers")
        if self.page_count is not None and any(
            page > self.page_count for page in missing_text_pages
        ):
            raise ValueError("missing_text_pages cannot exceed page_count")
        object.__setattr__(self, "missing_text_pages", missing_text_pages)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "company": self.company.to_dict(),
            "form_type": self.form_type,
            "accession_number": self.accession_number,
            "filing_date": self.filing_date.isoformat(),
            "report_date": self.report_date.isoformat() if self.report_date is not None else None,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date is not None else None
            ),
            "document_url": self.document_url,
            "source_url": self.source_url,
            "document_format": self.document_format.value,
            "retrieved_at": self.retrieved_at.isoformat(),
            "local_raw_path": self.local_raw_path,
            "local_text_path": self.local_text_path,
            "source": self.source.to_dict(),
            "chunk_ids": list(self.chunk_ids),
            "warnings": list(self.warnings),
            "extraction_status": (
                self.extraction_status.value if self.extraction_status is not None else None
            ),
            "extraction_method": (
                self.extraction_method.value if self.extraction_method is not None else None
            ),
            "page_count": self.page_count,
            "missing_text_pages": list(self.missing_text_pages),
        }


@dataclass(frozen=True, slots=True)
class FilingChunk:
    id: str
    filing_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    source_url: str
    accession_number: str
    form_type: str
    section_heading: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    source_region: DocumentRegion | None = None
    extraction_method: DocumentExtractionMethod | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "filing_id", _require_text("filing_id", self.filing_id))
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        object.__setattr__(self, "text", _require_text("text", self.text))
        if self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(
            self,
            "accession_number",
            _require_text("accession_number", self.accession_number),
        )
        object.__setattr__(self, "form_type", _require_text("form_type", self.form_type).upper())
        object.__setattr__(self, "section_heading", _optional_text(self.section_heading))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))
        if self.source_region is not None and not isinstance(self.source_region, DocumentRegion):
            raise ValueError("source_region must be a DocumentRegion")
        if self.extraction_method is not None:
            object.__setattr__(
                self,
                "extraction_method",
                DocumentExtractionMethod(self.extraction_method),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "filing_id": self.filing_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_heading": self.section_heading,
            "source_url": self.source_url,
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "metadata": dict(self.metadata),
            "source_region": (
                self.source_region.to_dict() if self.source_region is not None else None
            ),
            "extraction_method": (
                self.extraction_method.value if self.extraction_method is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class FilingIngestionResult:
    company: FilingCompany
    filings: tuple[FilingDocument, ...]
    chunks: tuple[FilingChunk, ...]
    source: FilingSource
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.company, FilingCompany):
            raise ValueError("company must be a FilingCompany")
        filings = tuple(self.filings)
        chunks = tuple(self.chunks)
        for index, filing in enumerate(filings):
            if not isinstance(filing, FilingDocument):
                raise ValueError(f"filings[{index}] must be a FilingDocument")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, FilingChunk):
                raise ValueError(f"chunks[{index}] must be a FilingChunk")
        if not isinstance(self.source, FilingSource):
            raise ValueError("source must be a FilingSource")
        object.__setattr__(self, "filings", filings)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "company": self.company.to_dict(),
            "filings": [filing.to_dict() for filing in self.filings],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "source": self.source.to_dict(),
            "warnings": list(self.warnings),
        }


class FilingProvider(Protocol):
    async def ingest_latest(
        self,
        company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult: ...


class FilingError(Exception):
    def __init__(
        self,
        *,
        code: FilingErrorCode | str,
        message: str,
        provider: str,
        retryable: bool = False,
    ) -> None:
        self.code = FilingErrorCode(code)
        self.message = _require_text("message", message)
        self.provider = _require_text("provider", provider)
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "filing_error",
            "code": self.code.value,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
        }


def _normalize_cik(value: str) -> str:
    text = _require_text("cik", value)
    if text.upper().startswith("CIK"):
        text = text[3:]
    digits = text.lstrip("0") or "0"
    if not digits.isdigit() or int(digits) <= 0:
        raise ValueError("cik must contain positive digits")
    if len(digits) > 10:
        raise ValueError("cik must be at most 10 digits")
    return digits


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
