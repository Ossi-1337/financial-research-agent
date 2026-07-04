from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from financial_research_agent.filings import (
    FilingCompany,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingStore,
    SECEDGARFilingProvider,
    build_chunks,
    detect_document_format,
    extract_document_text,
)

SUBMISSIONS_FIXTURE = {
    "cik": 320193,
    "name": "TEST TOOL OUTPUT APPLE INC.",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-25-000001", "0000320193-25-000000"],
            "filingDate": ["2026-01-31", "2025-10-31"],
            "reportDate": ["2025-12-31", "2025-09-30"],
            "form": ["10-K", "8-K"],
            "primaryDocument": ["aapl-20251231.htm", "aapl-8k.htm"],
        }
    },
}

HTML_DOCUMENT = b"""
<html>
  <head><title>ignored</title><style>.x{}</style></head>
  <body>
    <h1>Item 1. Business</h1>
    <p>TEST TOOL OUTPUT annual filing content.</p>
    <table><tr><td>Revenue table text</td></tr></table>
  </body>
</html>
"""


def test_filing_company_contract_is_immutable_and_normalizes_cik() -> None:
    company = FilingCompany(cik="CIK0000320193", legal_name="Test Company")

    assert company.cik == "320193"
    assert company.padded_cik == "0000320193"
    with pytest.raises(FrozenInstanceError):
        company.cik = "1"  # type: ignore[misc]


def test_document_format_detection_and_extraction() -> None:
    assert detect_document_format(document_name="filing.htm") == FilingDocumentFormat.HTML
    assert detect_document_format(document_name="filing.txt") == FilingDocumentFormat.TEXT
    assert detect_document_format(document_name="filing.pdf") == FilingDocumentFormat.PDF

    html_text = extract_document_text(HTML_DOCUMENT, FilingDocumentFormat.HTML)
    txt_text = extract_document_text(b"Line 1\r\n\r\nLine 2", FilingDocumentFormat.TEXT)

    assert "Item 1. Business" in html_text
    assert "Revenue table text" in html_text
    assert txt_text == "Line 1\n\nLine 2"
    with pytest.raises(FilingError) as exc_info:
        extract_document_text(b"%PDF", FilingDocumentFormat.PDF)
    assert exc_info.value.code == FilingErrorCode.UNSUPPORTED_FORMAT


def test_chunking_preserves_source_metadata_and_heading() -> None:
    text = "Item 1. Business\n" + ("alpha " * 300) + "\n\nItem 1A. Risk Factors\nbeta"

    chunks = build_chunks(
        filing_id="filing:fixture",
        text=text,
        source_url="https://example.invalid/filing.htm",
        accession_number="0000320193-25-000001",
        form_type="10-K",
        chunk_size=120,
        overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0].section_heading == "Item 1. Business"
    assert chunks[0].source_url == "https://example.invalid/filing.htm"
    assert chunks[0].metadata["overlap"] == "20"


def test_sec_edgar_provider_ingests_latest_html_filing(tmp_path) -> None:
    requests: list[str] = []
    provider = SECEDGARFilingProvider(
        raw_documents_dir=tmp_path / "raw",
        extracted_text_dir=tmp_path / "text",
        http_client=_sec_client(requests=requests),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    result = asyncio.run(
        provider.ingest_latest(
            FilingCompany(cik="0000320193", company_id="sec:cik:0000320193"),
            forms=("10-K",),
        )
    )

    filing = result.filings[0]
    assert result.company.legal_name == "TEST TOOL OUTPUT APPLE INC."
    assert result.source.provider == "sec-edgar"
    assert filing.form_type == "10-K"
    assert filing.document_url.endswith(
        "/Archives/edgar/data/320193/000032019325000001/aapl-20251231.htm"
    )
    assert filing.local_raw_path.endswith("aapl-20251231.htm")
    assert filing.local_text_path.endswith("aapl-20251231.txt")
    extracted_text = next((tmp_path / "text").rglob("*.txt")).read_text(encoding="utf-8")
    assert "TEST TOOL OUTPUT annual filing content" in extracted_text
    assert filing.chunk_ids == tuple(chunk.id for chunk in result.chunks)
    assert result.chunks[0].accession_number == "0000320193-25-000001"
    assert any("CIK0000320193.json" in request for request in requests)


def test_sec_edgar_provider_maps_rate_limit_and_malformed_payload() -> None:
    rate_limited = SECEDGARFilingProvider(
        http_client=_single_response_client(httpx.Response(429, json={"error": "slow down"})),
    )
    malformed = SECEDGARFilingProvider(
        http_client=_single_response_client(httpx.Response(200, json={"filings": {"recent": {}}})),
    )

    with pytest.raises(FilingError) as rate_error:
        asyncio.run(rate_limited.ingest_latest(FilingCompany(cik="320193")))
    with pytest.raises(FilingError) as malformed_error:
        asyncio.run(malformed.ingest_latest(FilingCompany(cik="320193")))

    assert rate_error.value.code == FilingErrorCode.RATE_LIMITED
    assert rate_error.value.retryable is True
    assert malformed_error.value.code == FilingErrorCode.MALFORMED_RESPONSE


def test_sec_edgar_provider_rejects_oversized_and_pdf_documents(tmp_path) -> None:
    oversized = SECEDGARFilingProvider(
        raw_documents_dir=tmp_path / "raw",
        extracted_text_dir=tmp_path / "text",
        max_document_bytes=10,
        http_client=_sec_client(document=HTML_DOCUMENT),
    )
    pdf_payload = {
        **SUBMISSIONS_FIXTURE,
        "filings": {
            "recent": {
                **SUBMISSIONS_FIXTURE["filings"]["recent"],
                "primaryDocument": ["aapl-20251231.pdf", "aapl-8k.htm"],
            }
        },
    }
    pdf = SECEDGARFilingProvider(
        raw_documents_dir=tmp_path / "raw-pdf",
        extracted_text_dir=tmp_path / "text-pdf",
        http_client=_sec_client(submissions=pdf_payload, document=b"%PDF"),
    )

    with pytest.raises(FilingError) as oversized_error:
        asyncio.run(oversized.ingest_latest(FilingCompany(cik="320193")))
    with pytest.raises(FilingError) as pdf_error:
        asyncio.run(pdf.ingest_latest(FilingCompany(cik="320193")))

    assert oversized_error.value.code == FilingErrorCode.DOCUMENT_TOO_LARGE
    assert pdf_error.value.code == FilingErrorCode.UNSUPPORTED_FORMAT


def test_sec_edgar_provider_rejects_malformed_content_length() -> None:
    provider = SECEDGARFilingProvider(
        http_client=_sec_client(document_headers={"content-length": "not-a-number"}),
    )

    with pytest.raises(FilingError) as exc_info:
        asyncio.run(provider.ingest_latest(FilingCompany(cik="320193")))

    assert exc_info.value.code == FilingErrorCode.MALFORMED_RESPONSE


def test_filing_store_persists_and_marks_stale_results(tmp_path) -> None:
    provider = SECEDGARFilingProvider(
        raw_documents_dir=tmp_path / "raw",
        extracted_text_dir=tmp_path / "text",
        http_client=_sec_client(),
        now=lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    result = asyncio.run(provider.ingest_latest(FilingCompany(cik="320193")))
    store = FilingStore(
        storage_path=tmp_path / "filings_index.json",
        stale_after=timedelta(days=1),
    )

    store.save_result(result)
    reloaded = FilingStore(
        storage_path=tmp_path / "filings_index.json",
        stale_after=timedelta(days=1),
    )
    fresh = reloaded.get_result(cik="0000320193", now=datetime(2026, 7, 1, tzinfo=UTC))
    stale = reloaded.get_result(cik="320193", now=datetime(2026, 7, 4, tzinfo=UTC))

    assert fresh is not None
    assert fresh.filings[0].form_type == "10-K"
    assert stale is not None
    assert "Stored filing documents are stale" in stale.warnings[-1]
    assert stale.source.freshness_warning == (
        "Stored filing documents are stale; refresh before relying on them."
    )


def _sec_client(
    *,
    requests: list[str] | None = None,
    submissions: object = SUBMISSIONS_FIXTURE,
    document: bytes = HTML_DOCUMENT,
    document_headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(str(request.url))
        if "submissions" in str(request.url):
            return httpx.Response(200, json=submissions)
        return httpx.Response(
            200,
            content=document,
            headers={"content-type": "text/html", **(document_headers or {})},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _single_response_client(response: httpx.Response) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: response))
