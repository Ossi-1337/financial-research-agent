from __future__ import annotations

import multiprocessing
import re
from collections.abc import Callable, Mapping
from multiprocessing.connection import Connection
from time import monotonic

from financial_research_agent.documents.contracts import (
    DocumentExtractionError,
    DocumentExtractionErrorCode,
    DocumentExtractionMethod,
    DocumentExtractionResult,
    DocumentExtractionStatus,
    DocumentFormat,
    DocumentPage,
    DocumentRegion,
    NormalizedDocument,
)

DEFAULT_MAX_DOCUMENT_BYTES = 50_000_000
DEFAULT_MAX_PAGES = 300
DEFAULT_MAX_EXTRACTED_CHARS = 2_000_000
DEFAULT_EXTRACTION_TIMEOUT_SECONDS = 120.0
MIN_PAGE_TEXT_CHARS = 20


class PDFDocumentExtractor:
    def __init__(
        self,
        *,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
        timeout_seconds: float = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
        _worker_target: Callable[..., None] | None = None,
    ) -> None:
        if max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be positive")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if max_extracted_chars <= 0:
            raise ValueError("max_extracted_chars must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.max_document_bytes = max_document_bytes
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars
        self.timeout_seconds = timeout_seconds
        self._worker_target = _worker_target or _pdf_worker

    def extract(
        self,
        content: bytes,
        *,
        content_type: str | None = None,
    ) -> DocumentExtractionResult:
        _validate_pdf_input(
            content,
            content_type=content_type,
            max_document_bytes=self.max_document_bytes,
        )
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=self._worker_target,
            args=(
                child,
                content,
                self.max_pages,
                self.max_extracted_chars,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        try:
            deadline = monotonic() + self.timeout_seconds
            while True:
                if parent.poll(min(0.05, max(0, deadline - monotonic()))):
                    payload = parent.recv()
                    break
                if not process.is_alive():
                    raise DocumentExtractionError(
                        code=DocumentExtractionErrorCode.EXTRACTION_FAILED,
                        message="PDF extraction worker stopped unexpectedly.",
                    )
                if monotonic() >= deadline:
                    _stop_process(process)
                    raise DocumentExtractionError(
                        code=DocumentExtractionErrorCode.TIMEOUT,
                        message="PDF extraction exceeded the configured timeout.",
                    )
        except EOFError as exc:
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.EXTRACTION_FAILED,
                message="PDF extraction worker stopped unexpectedly.",
            ) from exc
        finally:
            parent.close()
            process.join(timeout=1)
            if process.is_alive():
                _stop_process(process)
        if not isinstance(payload, Mapping):
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.EXTRACTION_FAILED,
                message="PDF extraction worker returned an invalid result.",
            )
        if payload.get("ok") is not True:
            raise DocumentExtractionError(
                code=str(payload.get("code", DocumentExtractionErrorCode.EXTRACTION_FAILED.value)),
                message=str(payload.get("message", "PDF extraction failed.")),
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.EXTRACTION_FAILED,
                message="PDF extraction worker returned no document.",
            )
        return DocumentExtractionResult.from_dict(result)


def _pdf_worker(
    connection: Connection,
    content: bytes,
    max_pages: int,
    max_extracted_chars: int,
) -> None:
    try:
        result = _extract_pdf_bytes(
            content,
            max_pages=max_pages,
            max_extracted_chars=max_extracted_chars,
        )
        connection.send({"ok": True, "result": result.to_dict()})
    except DocumentExtractionError as exc:
        connection.send({"ok": False, "code": exc.code.value, "message": exc.message})
    except Exception:
        connection.send(
            {
                "ok": False,
                "code": DocumentExtractionErrorCode.EXTRACTION_FAILED.value,
                "message": "PDF extraction failed safely.",
            }
        )
    finally:
        connection.close()


def _extract_pdf_bytes(
    content: bytes,
    *,
    max_pages: int,
    max_extracted_chars: int,
) -> DocumentExtractionResult:
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(content)
    except Exception as exc:
        message = str(exc).lower()
        if "password" in message or "security" in message:
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.ENCRYPTED_DOCUMENT,
                message="Encrypted PDF documents are not supported.",
            ) from exc
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.INVALID_DOCUMENT,
            message="PDF document is malformed or unreadable.",
        ) from exc

    try:
        page_count = len(document)
        if page_count <= 0:
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.INVALID_DOCUMENT,
                message="PDF document contains no pages.",
            )
        if page_count > max_pages:
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.PAGE_LIMIT_EXCEEDED,
                message="PDF document exceeds the configured page limit.",
            )
        raw_pages = [_extract_page(document, index) for index in range(page_count)]
    finally:
        document.close()

    missing_pages = tuple(
        page_number
        for page_number, text, _regions in raw_pages
        if len(re.sub(r"\s+", "", text)) < MIN_PAGE_TEXT_CHARS
    )
    text_pages = len(raw_pages) - len(missing_pages)
    if text_pages == 0:
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.OCR_REQUIRED,
            message="PDF contains no usable native text; OCR is required.",
        )

    normalized_parts: list[str] = []
    pages: list[DocumentPage] = []
    cursor = 0
    for page_number, text, regions in raw_pages:
        if normalized_parts:
            separator = "\n\f\n"
            normalized_parts.append(separator)
            cursor += len(separator)
        page_start = cursor
        normalized_parts.append(text)
        cursor += len(text)
        page_warnings = ("native_text_missing",) if page_number in missing_pages else ()
        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text,
                char_start=page_start,
                char_end=cursor,
                regions=regions,
                warnings=page_warnings,
            )
        )
        if cursor > max_extracted_chars:
            raise DocumentExtractionError(
                code=DocumentExtractionErrorCode.EXTRACTED_TEXT_TOO_LARGE,
                message="PDF extracted text exceeds the configured character limit.",
            )

    status = (
        DocumentExtractionStatus.PARTIAL if missing_pages else DocumentExtractionStatus.COMPLETE
    )
    warnings = (
        (f"Native text was unavailable on pages: {', '.join(map(str, missing_pages))}.",)
        if missing_pages
        else ()
    )
    normalized = NormalizedDocument(
        document_format=DocumentFormat.PDF,
        extraction_method=DocumentExtractionMethod.PDF_NATIVE_TEXT,
        status=status,
        text="".join(normalized_parts),
        pages=tuple(pages),
        warnings=warnings,
    )
    return DocumentExtractionResult(document=normalized, missing_text_pages=missing_pages)


def _extract_page(document, page_index: int) -> tuple[int, str, tuple[DocumentRegion, ...]]:
    page = document[page_index]
    try:
        width, height = page.get_size()
        text_page = page.get_textpage()
        try:
            text = _normalize_pdf_text(text_page.get_text_bounded())
            regions = _page_regions(
                text_page,
                page_number=page_index + 1,
                width=width,
                height=height,
            )
        finally:
            text_page.close()
    finally:
        page.close()
    return page_index + 1, text, regions


def _page_regions(
    text_page,
    *,
    page_number: int,
    width: float,
    height: float,
) -> tuple[DocumentRegion, ...]:
    if width <= 0 or height <= 0:
        return ()
    try:
        count = text_page.count_rects()
        rectangles = [text_page.get_rect(index) for index in range(count)]
    except Exception:
        return ()
    if not rectangles:
        return ()
    left = min(rectangle[0] for rectangle in rectangles)
    bottom = min(rectangle[1] for rectangle in rectangles)
    right = max(rectangle[2] for rectangle in rectangles)
    top = max(rectangle[3] for rectangle in rectangles)
    normalized_left = _clamp(left / width)
    normalized_top = _clamp(1 - (top / height))
    normalized_right = _clamp(right / width)
    normalized_bottom = _clamp(1 - (bottom / height))
    if normalized_right <= normalized_left or normalized_bottom <= normalized_top:
        return ()
    return (
        DocumentRegion(
            page_number=page_number,
            left=normalized_left,
            top=normalized_top,
            right=normalized_right,
            bottom=normalized_bottom,
        ),
    )


def _validate_pdf_input(
    content: bytes,
    *,
    content_type: str | None,
    max_document_bytes: int,
) -> None:
    if not isinstance(content, bytes):
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.INVALID_DOCUMENT,
            message="PDF content must be bytes.",
        )
    if len(content) > max_document_bytes:
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.DOCUMENT_TOO_LARGE,
            message="PDF document exceeds the configured size limit.",
        )
    if not content.startswith(b"%PDF-"):
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.INVALID_DOCUMENT,
            message="Document does not have a valid PDF signature.",
        )
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type and media_type != "application/pdf":
        raise DocumentExtractionError(
            code=DocumentExtractionErrorCode.INVALID_DOCUMENT,
            message="Document content type is not application/pdf.",
        )


def _normalize_pdf_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stop_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
