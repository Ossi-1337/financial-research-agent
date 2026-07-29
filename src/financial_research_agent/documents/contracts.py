from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DocumentFormat(StrEnum):
    HTML = "html"
    TEXT = "text"
    PDF = "pdf"


class DocumentExtractionMethod(StrEnum):
    HTML_PARSER = "html_parser"
    PLAIN_TEXT = "plain_text"
    PDF_NATIVE_TEXT = "pdf_native_text"


class DocumentExtractionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    OCR_REQUIRED = "ocr_required"


class DocumentExtractionErrorCode(StrEnum):
    INVALID_DOCUMENT = "invalid_document"
    DOCUMENT_TOO_LARGE = "document_too_large"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    EXTRACTED_TEXT_TOO_LARGE = "extracted_text_too_large"
    ENCRYPTED_DOCUMENT = "encrypted_document"
    OCR_REQUIRED = "ocr_required"
    TIMEOUT = "timeout"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass(frozen=True, slots=True)
class DocumentRegion:
    page_number: int
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.page_number <= 0:
            raise ValueError("page_number must be positive")
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DocumentRegion:
        return cls(
            page_number=int(value["page_number"]),
            left=float(value["left"]),
            top=float(value["top"]),
            right=float(value["right"]),
            bottom=float(value["bottom"]),
        )


@dataclass(frozen=True, slots=True)
class DocumentPage:
    page_number: int
    text: str
    char_start: int
    char_end: int
    regions: tuple[DocumentRegion, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.page_number <= 0:
            raise ValueError("page_number must be positive")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end < self.char_start:
            raise ValueError("char_end must not precede char_start")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("page character offsets must match text length")
        regions = tuple(self.regions)
        if any(region.page_number != self.page_number for region in regions):
            raise ValueError("page regions must reference their owning page")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "regions": [region.to_dict() for region in self.regions],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DocumentPage:
        return cls(
            page_number=int(value["page_number"]),
            text=str(value["text"]),
            char_start=int(value["char_start"]),
            char_end=int(value["char_end"]),
            regions=tuple(
                DocumentRegion.from_dict(region)
                for region in value.get("regions", [])
                if isinstance(region, dict)
            ),
            warnings=tuple(str(item) for item in value.get("warnings", [])),
        )


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    document_format: DocumentFormat
    extraction_method: DocumentExtractionMethod
    status: DocumentExtractionStatus
    text: str
    pages: tuple[DocumentPage, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_format", DocumentFormat(self.document_format))
        object.__setattr__(
            self,
            "extraction_method",
            DocumentExtractionMethod(self.extraction_method),
        )
        object.__setattr__(self, "status", DocumentExtractionStatus(self.status))
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        pages = tuple(self.pages)
        if not pages:
            raise ValueError("pages must not be empty")
        if tuple(page.page_number for page in pages) != tuple(range(1, len(pages) + 1)):
            raise ValueError("pages must use contiguous 1-based numbering")
        for page in pages:
            if self.text[page.char_start : page.char_end] != page.text:
                raise ValueError("page offsets must reference normalized document text")
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "document_format": self.document_format.value,
            "extraction_method": self.extraction_method.value,
            "status": self.status.value,
            "text": self.text,
            "pages": [page.to_dict() for page in self.pages],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> NormalizedDocument:
        return cls(
            document_format=DocumentFormat(str(value["document_format"])),
            extraction_method=DocumentExtractionMethod(str(value["extraction_method"])),
            status=DocumentExtractionStatus(str(value["status"])),
            text=str(value["text"]),
            pages=tuple(
                DocumentPage.from_dict(page)
                for page in value.get("pages", [])
                if isinstance(page, dict)
            ),
            warnings=tuple(str(item) for item in value.get("warnings", [])),
        )


@dataclass(frozen=True, slots=True)
class DocumentExtractionResult:
    document: NormalizedDocument
    missing_text_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.document, NormalizedDocument):
            raise ValueError("document must be a NormalizedDocument")
        pages = tuple(self.missing_text_pages)
        if any(page <= 0 or page > len(self.document.pages) for page in pages):
            raise ValueError("missing_text_pages contains an invalid page")
        object.__setattr__(self, "missing_text_pages", pages)

    def to_dict(self) -> dict[str, object]:
        return {
            "document": self.document.to_dict(),
            "missing_text_pages": list(self.missing_text_pages),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DocumentExtractionResult:
        document = value.get("document")
        if not isinstance(document, dict):
            raise ValueError("document is required")
        return cls(
            document=NormalizedDocument.from_dict(document),
            missing_text_pages=tuple(int(page) for page in value.get("missing_text_pages", [])),
        )


class DocumentExtractor(Protocol):
    def extract(
        self,
        content: bytes,
        *,
        content_type: str | None = None,
    ) -> DocumentExtractionResult: ...


class DocumentExtractionError(Exception):
    def __init__(
        self,
        *,
        code: DocumentExtractionErrorCode | str,
        message: str,
    ) -> None:
        self.code = DocumentExtractionErrorCode(code)
        self.message = _require_text("message", message)
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str]:
        return {
            "error": "document_extraction_error",
            "code": self.code.value,
            "message": self.message,
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))
