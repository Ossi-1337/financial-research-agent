from __future__ import annotations

import html
import re
from dataclasses import replace
from html.parser import HTMLParser

from financial_research_agent.documents import NormalizedDocument
from financial_research_agent.filings.contracts import (
    FilingChunk,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingProviderName,
)

DEFAULT_CHUNK_SIZE = 4_000
DEFAULT_CHUNK_OVERLAP = 250

_BLOCK_TAGS = {
    "address",
    "article",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
_IGNORED_TAGS = {"script", "style", "title"}
_HEADING_PATTERN = re.compile(r"^(item\s+\d+[a-z]?\.?|part\s+[ivx]+\.?)\b", re.IGNORECASE)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        lowered = tag.lower()
        if lowered in _IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if lowered in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _IGNORED_TAGS and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if lowered in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        self.parts.append(data)

    def text(self) -> str:
        return normalize_extracted_text("".join(self.parts))


def detect_document_format(
    *,
    document_name: str,
    content_type: str | None = None,
) -> FilingDocumentFormat:
    name = document_name.strip().lower()
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if name.endswith((".htm", ".html")):
        return FilingDocumentFormat.HTML
    if name.endswith(".txt"):
        return FilingDocumentFormat.TEXT
    if name.endswith(".pdf"):
        return FilingDocumentFormat.PDF
    if media_type in {"text/html", "application/xhtml+xml"}:
        return FilingDocumentFormat.HTML
    if media_type in {"text/plain", "application/text"}:
        return FilingDocumentFormat.TEXT
    if media_type == "application/pdf":
        return FilingDocumentFormat.PDF
    return FilingDocumentFormat.UNSUPPORTED


def extract_document_text(content: bytes, document_format: FilingDocumentFormat | str) -> str:
    selected_format = FilingDocumentFormat(document_format)
    if selected_format == FilingDocumentFormat.PDF:
        raise FilingError(
            code=FilingErrorCode.UNSUPPORTED_FORMAT,
            message="Use PDFDocumentExtractor for PDF text and page metadata.",
            provider=FilingProviderName.SEC_EDGAR.value,
        )
    if selected_format == FilingDocumentFormat.UNSUPPORTED:
        raise FilingError(
            code=FilingErrorCode.UNSUPPORTED_FORMAT,
            message="Unsupported filing document format.",
            provider=FilingProviderName.SEC_EDGAR.value,
        )
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = content.decode("latin-1", errors="replace")
    if selected_format == FilingDocumentFormat.HTML:
        parser = _HTMLTextExtractor()
        try:
            parser.feed(decoded)
            parser.close()
        except Exception as exc:
            raise FilingError(
                code=FilingErrorCode.EXTRACTION_FAILED,
                message="Could not extract text from SEC HTML filing document.",
                provider=FilingProviderName.SEC_EDGAR.value,
            ) from exc
        text = parser.text()
    else:
        text = normalize_extracted_text(html.unescape(decoded))
    if text == "":
        raise FilingError(
            code=FilingErrorCode.EXTRACTION_FAILED,
            message="Filing document did not contain extractable text.",
            provider=FilingProviderName.SEC_EDGAR.value,
        )
    return text


def normalize_extracted_text(value: str) -> str:
    text = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    normalized: list[str] = []
    previous_blank = True
    for line in lines:
        if line == "":
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def build_chunks(
    *,
    filing_id: str,
    text: str,
    source_url: str,
    accession_number: str,
    form_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[FilingChunk, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if text.strip() == "":
        return ()
    chunks: list[FilingChunk] = []
    start = 0
    index = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = _chunk_end(text, start, target_end)
        chunk_text = text[start:end].strip()
        if chunk_text:
            char_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            char_end = char_start + len(chunk_text)
            chunks.append(
                FilingChunk(
                    id=f"{filing_id}:chunk:{index}",
                    filing_id=filing_id,
                    chunk_index=index,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    section_heading=_section_heading_for_text(text[:char_start], chunk_text),
                    source_url=source_url,
                    accession_number=accession_number,
                    form_type=form_type,
                    metadata={
                        "chunk_size": str(chunk_size),
                        "overlap": str(overlap),
                    },
                )
            )
            index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return tuple(chunks)


def build_document_chunks(
    *,
    filing_id: str,
    document: NormalizedDocument,
    source_url: str,
    accession_number: str,
    form_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[FilingChunk, ...]:
    chunks: list[FilingChunk] = []
    for page in document.pages:
        page_chunks = build_chunks(
            filing_id=filing_id,
            text=page.text,
            source_url=source_url,
            accession_number=accession_number,
            form_type=form_type,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for page_chunk in page_chunks:
            chunk_index = len(chunks)
            chunks.append(
                replace(
                    page_chunk,
                    id=f"{filing_id}:chunk:{chunk_index}",
                    chunk_index=chunk_index,
                    char_start=page.char_start + page_chunk.char_start,
                    char_end=page.char_start + page_chunk.char_end,
                    source_region=page.regions[0] if page.regions else None,
                    extraction_method=document.extraction_method,
                    metadata={
                        **dict(page_chunk.metadata),
                        "page_number": str(page.page_number),
                        "extraction_method": document.extraction_method.value,
                    },
                )
            )
    return tuple(chunks)


def _chunk_end(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    paragraph_break = text.rfind("\n\n", start, target_end)
    if paragraph_break > start:
        return paragraph_break
    line_break = text.rfind("\n", start, target_end)
    if line_break > start:
        return line_break
    space = text.rfind(" ", start, target_end)
    if space > start:
        return space
    return target_end


def _section_heading_for_text(prefix: str, chunk_text: str) -> str | None:
    for line in reversed(prefix.splitlines()[-20:] + chunk_text.splitlines()[:3]):
        candidate = line.strip()
        if candidate and _HEADING_PATTERN.match(candidate):
            return candidate[:160]
    return None
