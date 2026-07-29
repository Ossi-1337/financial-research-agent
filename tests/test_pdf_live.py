from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from financial_research_agent.documents import (
    DocumentExtractionStatus,
    PDFDocumentExtractor,
)

NOVO_ANNUAL_REPORT_2025_URL = (
    "https://www.novonordisk.com/content/dam/nncorp/global/en/investors/"
    "irmaterial/annual_report/2026/novo-nordisk-annual-report-2025.pdf"
)


@pytest.mark.skipif(
    os.environ.get("FRA_PDF_LIVE_SMOKE_TEST") != "1",
    reason="Set FRA_PDF_LIVE_SMOKE_TEST=1 to run the official PDF smoke test.",
)
def test_live_novo_annual_report_pdf_extraction(tmp_path: Path) -> None:
    response = httpx.get(NOVO_ANNUAL_REPORT_2025_URL, timeout=120, follow_redirects=True)
    response.raise_for_status()
    local_pdf = tmp_path / "novo-nordisk-annual-report-2025.pdf"
    local_pdf.write_bytes(response.content)

    result = PDFDocumentExtractor(timeout_seconds=120).extract(
        local_pdf.read_bytes(),
        content_type=response.headers.get("content-type"),
    )
    text = result.document.text.casefold()

    assert result.document.status in {
        DocumentExtractionStatus.COMPLETE,
        DocumentExtractionStatus.PARTIAL,
    }
    assert len(result.document.pages) >= 100
    assert "novo nordisk" in text
    assert any(term in text for term in ("revenue", "operating profit", "cash flow"))
    assert all(
        0 <= value <= 1
        for page in result.document.pages
        for region in page.regions
        for value in (region.left, region.top, region.right, region.bottom)
    )
