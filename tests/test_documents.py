from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from financial_research_agent.documents import (
    DocumentExtractionError,
    DocumentExtractionErrorCode,
    DocumentExtractionMethod,
    DocumentExtractionStatus,
    DocumentFormat,
    DocumentPage,
    DocumentRegion,
    NormalizedDocument,
    PDFDocumentExtractor,
)


def test_document_contracts_are_immutable_and_validate_regions() -> None:
    region = DocumentRegion(page_number=1, left=0.1, top=0.2, right=0.9, bottom=0.8)
    page = DocumentPage(
        page_number=1,
        text="Revenue increased.",
        char_start=0,
        char_end=18,
        regions=(region,),
    )
    document = NormalizedDocument(
        document_format=DocumentFormat.PDF,
        extraction_method=DocumentExtractionMethod.PDF_NATIVE_TEXT,
        status=DocumentExtractionStatus.COMPLETE,
        text=page.text,
        pages=(page,),
    )

    assert NormalizedDocument.from_dict(document.to_dict()) == document
    with pytest.raises(AttributeError):
        region.left = 0.2  # type: ignore[misc]
    with pytest.raises(ValueError, match="between 0 and 1"):
        DocumentRegion(page_number=1, left=-0.1, top=0, right=1, bottom=1)


def test_pdf_extractor_preserves_pages_unicode_and_coordinates() -> None:
    content = _pdf_bytes(
        [
            "Årsrapport: omsætning og likviditet.",
            "Cash flow, liabilities, and equity.",
        ],
        unicode_font=True,
    )

    result = PDFDocumentExtractor(timeout_seconds=20).extract(
        content,
        content_type="application/pdf",
    )

    assert result.document.status == DocumentExtractionStatus.COMPLETE
    assert len(result.document.pages) == 2
    assert "Årsrapport" in result.document.pages[0].text
    assert result.document.pages[0].char_end <= result.document.pages[1].char_start
    assert all(page.regions for page in result.document.pages)
    assert all(
        0 <= value <= 1
        for page in result.document.pages
        for region in page.regions
        for value in (region.left, region.top, region.right, region.bottom)
    )


def test_pdf_extractor_marks_mixed_native_text_as_partial() -> None:
    result = PDFDocumentExtractor(timeout_seconds=20).extract(
        _pdf_bytes(["Annual report revenue and operating profit.", None])
    )

    assert result.document.status == DocumentExtractionStatus.PARTIAL
    assert result.missing_text_pages == (2,)
    assert result.document.pages[1].warnings == ("native_text_missing",)


def test_pdf_extractor_requires_ocr_for_blank_document() -> None:
    with pytest.raises(DocumentExtractionError) as raised:
        PDFDocumentExtractor(timeout_seconds=20).extract(_pdf_bytes([None, None]))

    assert raised.value.code == DocumentExtractionErrorCode.OCR_REQUIRED


def test_pdf_extractor_rejects_encrypted_document() -> None:
    with pytest.raises(DocumentExtractionError) as raised:
        PDFDocumentExtractor(timeout_seconds=20).extract(
            _pdf_bytes(["Annual report financial text."], encrypted=True)
        )

    assert raised.value.code == DocumentExtractionErrorCode.ENCRYPTED_DOCUMENT


@pytest.mark.parametrize(
    ("content", "content_type", "code"),
    [
        (b"not-a-pdf", "application/pdf", DocumentExtractionErrorCode.INVALID_DOCUMENT),
        (b"%PDF-malformed", "application/pdf", DocumentExtractionErrorCode.INVALID_DOCUMENT),
    ],
)
def test_pdf_extractor_rejects_invalid_input(
    content: bytes,
    content_type: str,
    code: DocumentExtractionErrorCode,
) -> None:
    with pytest.raises(DocumentExtractionError) as raised:
        PDFDocumentExtractor(timeout_seconds=20).extract(content, content_type=content_type)

    assert raised.value.code == code


def test_pdf_extractor_rejects_non_pdf_content_type() -> None:
    with pytest.raises(DocumentExtractionError) as raised:
        PDFDocumentExtractor(timeout_seconds=20).extract(
            _pdf_bytes(["Annual financial report text."]),
            content_type="text/plain",
        )

    assert raised.value.code == DocumentExtractionErrorCode.INVALID_DOCUMENT


def test_pdf_extractor_enforces_size_page_and_character_limits() -> None:
    content = _pdf_bytes(["Annual report financial text.", "Balance sheet financial text."])

    with pytest.raises(DocumentExtractionError) as size_error:
        PDFDocumentExtractor(max_document_bytes=10).extract(content)
    with pytest.raises(DocumentExtractionError) as page_error:
        PDFDocumentExtractor(max_pages=1, timeout_seconds=20).extract(content)
    with pytest.raises(DocumentExtractionError) as char_error:
        PDFDocumentExtractor(max_extracted_chars=10, timeout_seconds=20).extract(content)

    assert size_error.value.code == DocumentExtractionErrorCode.DOCUMENT_TOO_LARGE
    assert page_error.value.code == DocumentExtractionErrorCode.PAGE_LIMIT_EXCEEDED
    assert char_error.value.code == DocumentExtractionErrorCode.EXTRACTED_TEXT_TOO_LARGE


def test_pdf_worker_timeout_and_next_extraction_use_fresh_process() -> None:
    content = _pdf_bytes(["Annual report financial text."])
    extractor = PDFDocumentExtractor(timeout_seconds=0.05, _worker_target=_slow_worker)

    with pytest.raises(DocumentExtractionError) as raised:
        extractor.extract(content)
    assert raised.value.code == DocumentExtractionErrorCode.TIMEOUT

    extractor.timeout_seconds = 5
    extractor._worker_target = _crash_worker
    with pytest.raises(DocumentExtractionError) as crashed:
        extractor.extract(content)
    assert crashed.value.code == DocumentExtractionErrorCode.EXTRACTION_FAILED

    extractor._worker_target = _normal_worker
    extractor.timeout_seconds = 20
    assert extractor.extract(content).document.status == DocumentExtractionStatus.COMPLETE


def _pdf_bytes(
    pages: list[str | None],
    *,
    unicode_font: bool = False,
    encrypted: bool = False,
) -> bytes:
    output = BytesIO()
    canvas = Canvas(output, encrypt="test-password" if encrypted else None)
    font_name = "Helvetica"
    if unicode_font:
        font_name = "TestNotoSans"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            font_path = (
                Path(__file__).parents[1]
                / "src"
                / "financial_research_agent"
                / "report_exports"
                / "assets"
                / "NotoSans-Regular.ttf"
            )
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    for text in pages:
        if text:
            canvas.setFont(font_name, 12)
            canvas.drawString(72, 720, text)
        canvas.showPage()
    canvas.save()
    return output.getvalue()


def _slow_worker(*_args) -> None:
    time.sleep(10)


def _crash_worker(*_args) -> None:
    raise RuntimeError("fixture worker crash")


def _normal_worker(connection, content, max_pages, max_extracted_chars) -> None:
    from financial_research_agent.documents.pdf import _pdf_worker

    _pdf_worker(connection, content, max_pages, max_extracted_chars)
