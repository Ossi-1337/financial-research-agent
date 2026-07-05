from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from financial_research_agent.filings.contracts import (
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingIngestionResult,
    FilingProviderName,
    FilingSource,
)
from financial_research_agent.filings.extraction import (
    build_chunks,
    detect_document_format,
    extract_document_text,
)

SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_FILING_PROVIDER = FilingProviderName.SEC_EDGAR.value
SEC_FILING_PROVIDER_STATUS = "official"
SEC_FILING_ATTRIBUTION = "U.S. Securities and Exchange Commission EDGAR filings"
DEFAULT_MAX_DOCUMENT_BYTES = 8_000_000
DEFAULT_SUPPORTED_FORMS = ("10-K", "10-Q")
SEC_EXTRACTION_WARNING = (
    "SEC HTML/TXT extraction uses a lightweight local parser; full filing presentation, "
    "inline XBRL tables, and PDFs are deferred."
)
SEC_LOCAL_CACHE_WARNING = (
    "Downloaded filing documents are stored for local research cache only; do not "
    "redistribute cached issuer documents from the app."
)


class SECEDGARFilingProvider:
    def __init__(
        self,
        *,
        submissions_base_url: str = SEC_SUBMISSIONS_BASE_URL,
        archives_base_url: str = SEC_ARCHIVES_BASE_URL,
        raw_documents_dir: Path | None = None,
        extracted_text_dir: Path | None = None,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        user_agent: str = (
            "financial-research-agent/0.1 local-research contact@financial-research-agent.local"
        ),
        http_client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be positive")
        self.submissions_base_url = _require_text(
            "submissions_base_url", submissions_base_url
        ).rstrip("/")
        self.archives_base_url = _require_text("archives_base_url", archives_base_url).rstrip("/")
        self.raw_documents_dir = raw_documents_dir
        self.extracted_text_dir = extracted_text_dir
        self.max_document_bytes = max_document_bytes
        self.user_agent = _require_text("user_agent", user_agent)
        self._http_client = http_client
        self._now = now or (lambda: datetime.now(UTC))

    async def ingest_latest(
        self,
        company: FilingCompany,
        *,
        forms: tuple[str, ...] = DEFAULT_SUPPORTED_FORMS,
        limit: int = 1,
    ) -> FilingIngestionResult:
        selected_forms = _normalize_forms(forms)
        if limit <= 0 or limit > 10:
            raise FilingError(
                code=FilingErrorCode.INVALID_REQUEST,
                message="limit must be between 1 and 10",
                provider=SEC_FILING_PROVIDER,
            )
        retrieved_at = self._now()
        submissions_url = self._submissions_url(company)
        payload = await self._get_json(submissions_url)
        company = _company_from_payload(payload, company)
        candidates = _submission_records(payload, selected_forms)
        if not candidates:
            raise FilingError(
                code=FilingErrorCode.NOT_FOUND,
                message=(
                    f"No SEC submissions found for CIK {company.padded_cik} "
                    f"and forms {', '.join(selected_forms)}."
                ),
                provider=SEC_FILING_PROVIDER,
            )
        source = FilingSource(
            provider=SEC_FILING_PROVIDER,
            provider_status=SEC_FILING_PROVIDER_STATUS,
            source_url=submissions_url,
            retrieved_at=retrieved_at,
            data_as_of=max(record.filing_date for record in candidates),
            attribution=SEC_FILING_ATTRIBUTION,
        )
        filings: list[FilingDocument] = []
        chunks = []
        warnings = [SEC_EXTRACTION_WARNING, SEC_LOCAL_CACHE_WARNING]
        for record in candidates[:limit]:
            filing, filing_chunks = await self._ingest_record(company, record, retrieved_at)
            filings.append(filing)
            chunks.extend(filing_chunks)
        return FilingIngestionResult(
            company=company,
            filings=tuple(filings),
            chunks=tuple(chunks),
            source=source,
            warnings=tuple(warnings),
        )

    async def _ingest_record(
        self,
        company: FilingCompany,
        record: _SECSubmissionRecord,
        retrieved_at: datetime,
    ):
        document_url = self._document_url(company, record)
        content, content_type = await self._get_document(document_url)
        document_format = detect_document_format(
            document_name=record.primary_document,
            content_type=content_type,
        )
        text = extract_document_text(content, document_format)
        filing_id = f"sec:filing:{company.padded_cik}:{record.accession_number}"
        filing_source = FilingSource(
            provider=SEC_FILING_PROVIDER,
            provider_status=SEC_FILING_PROVIDER_STATUS,
            source_url=document_url,
            retrieved_at=retrieved_at,
            data_as_of=record.filing_date,
            attribution=SEC_FILING_ATTRIBUTION,
        )
        raw_path, text_path = self._write_document_files(
            company=company,
            record=record,
            content=content,
            text=text,
        )
        filing_chunks = build_chunks(
            filing_id=filing_id,
            text=text,
            source_url=document_url,
            accession_number=record.accession_number,
            form_type=record.form_type,
        )
        filing = FilingDocument(
            id=filing_id,
            company=company,
            form_type=record.form_type,
            accession_number=record.accession_number,
            filing_date=record.filing_date,
            report_date=record.report_date,
            publication_date=record.filing_date,
            document_url=document_url,
            source_url=self._submissions_url(company),
            document_format=document_format,
            retrieved_at=retrieved_at,
            local_raw_path=str(raw_path),
            local_text_path=str(text_path),
            source=filing_source,
            chunk_ids=tuple(chunk.id for chunk in filing_chunks),
            warnings=(SEC_EXTRACTION_WARNING,),
        )
        return filing, filing_chunks

    async def _get_json(self, url: str) -> Mapping[str, Any]:
        response = await self._get(url)
        try:
            payload = response.json()
        except ValueError as exc:
            raise FilingError(
                code=FilingErrorCode.MALFORMED_RESPONSE,
                message="SEC submissions source returned malformed JSON.",
                provider=SEC_FILING_PROVIDER,
            ) from exc
        if not isinstance(payload, Mapping):
            raise FilingError(
                code=FilingErrorCode.MALFORMED_RESPONSE,
                message="SEC submissions payload must be a JSON object.",
                provider=SEC_FILING_PROVIDER,
            )
        return payload

    async def _get_document(self, url: str) -> tuple[bytes, str | None]:
        response = await self._get(url)
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise FilingError(
                    code=FilingErrorCode.MALFORMED_RESPONSE,
                    message="SEC filing document returned an invalid content-length header.",
                    provider=SEC_FILING_PROVIDER,
                ) from exc
            if declared_size > self.max_document_bytes:
                raise FilingError(
                    code=FilingErrorCode.DOCUMENT_TOO_LARGE,
                    message="SEC filing document exceeds configured max document size.",
                    provider=SEC_FILING_PROVIDER,
                )
        content = response.content
        if len(content) > self.max_document_bytes:
            raise FilingError(
                code=FilingErrorCode.DOCUMENT_TOO_LARGE,
                message="SEC filing document exceeds configured max document size.",
                provider=SEC_FILING_PROVIDER,
            )
        return content, response.headers.get("content-type")

    async def _get(self, url: str) -> httpx.Response:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json,text/html,text/plain,*/*",
        }
        try:
            if self._http_client is None:
                async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                    response = await client.get(url)
            else:
                response = await self._http_client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise FilingError(
                code=FilingErrorCode.TIMEOUT,
                message="Timed out while fetching SEC filing data.",
                provider=SEC_FILING_PROVIDER,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise FilingError(
                code=FilingErrorCode.PROVIDER_UNAVAILABLE,
                message=f"SEC filing source is unavailable: {exc}",
                provider=SEC_FILING_PROVIDER,
                retryable=True,
            ) from exc
        _raise_for_status(response)
        return response

    def _write_document_files(
        self,
        *,
        company: FilingCompany,
        record: _SECSubmissionRecord,
        content: bytes,
        text: str,
    ) -> tuple[Path, Path]:
        raw_root = (
            self.raw_documents_dir or Path.cwd() / ".financial-research-agent" / "filings" / "raw"
        )
        text_root = (
            self.extracted_text_dir or Path.cwd() / ".financial-research-agent" / "filings" / "text"
        )
        accession_dir = _accession_no_dashes(record.accession_number)
        raw_dir = raw_root / company.padded_cik / accession_dir
        text_dir = text_root / company.padded_cik / accession_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / _safe_file_name(record.primary_document)
        text_path = text_dir / f"{Path(_safe_file_name(record.primary_document)).stem}.txt"
        raw_path.write_bytes(content)
        text_path.write_text(text, encoding="utf-8")
        return raw_path, text_path

    def _submissions_url(self, company: FilingCompany) -> str:
        return f"{self.submissions_base_url}/CIK{company.padded_cik}.json"

    def _document_url(self, company: FilingCompany, record: _SECSubmissionRecord) -> str:
        return (
            f"{self.archives_base_url}/{int(company.cik)}/"
            f"{_accession_no_dashes(record.accession_number)}/"
            f"{_safe_file_name(record.primary_document)}"
        )


class _SECSubmissionRecord:
    def __init__(
        self,
        *,
        accession_number: str,
        form_type: str,
        filing_date: date,
        report_date: date | None,
        primary_document: str,
    ) -> None:
        self.accession_number = _require_text("accession_number", accession_number)
        self.form_type = _require_text("form_type", form_type).upper()
        self.filing_date = filing_date
        self.report_date = report_date
        self.primary_document = _require_text("primary_document", primary_document)


def _submission_records(
    payload: Mapping[str, Any],
    forms: tuple[str, ...],
) -> tuple[_SECSubmissionRecord, ...]:
    filings = payload.get("filings")
    if not isinstance(filings, Mapping):
        raise FilingError(
            code=FilingErrorCode.MALFORMED_RESPONSE,
            message="SEC submissions payload does not include filings metadata.",
            provider=SEC_FILING_PROVIDER,
        )
    recent = filings.get("recent")
    if not isinstance(recent, Mapping):
        raise FilingError(
            code=FilingErrorCode.MALFORMED_RESPONSE,
            message="SEC submissions payload does not include recent filings.",
            provider=SEC_FILING_PROVIDER,
        )
    rows = _recent_rows(recent)
    records = [
        row
        for row in rows
        if row.form_type in forms
        and detect_document_format(document_name=row.primary_document)
        in {FilingDocumentFormat.HTML, FilingDocumentFormat.TEXT, FilingDocumentFormat.PDF}
    ]
    records.sort(key=lambda record: (record.filing_date, record.accession_number), reverse=True)
    return tuple(records)


def _recent_rows(recent: Mapping[str, Any]) -> tuple[_SECSubmissionRecord, ...]:
    forms = _column(recent, "form")
    accessions = _column(recent, "accessionNumber")
    filing_dates = _column(recent, "filingDate")
    report_dates = _column(recent, "reportDate")
    primary_documents = _column(recent, "primaryDocument")
    lengths = {
        len(forms),
        len(accessions),
        len(filing_dates),
        len(report_dates),
        len(primary_documents),
    }
    if len(lengths) != 1:
        raise FilingError(
            code=FilingErrorCode.MALFORMED_RESPONSE,
            message="SEC submissions recent columns have inconsistent lengths.",
            provider=SEC_FILING_PROVIDER,
        )
    records: list[_SECSubmissionRecord] = []
    for index, form in enumerate(forms):
        primary_document = str(primary_documents[index]).strip()
        if primary_document == "":
            continue
        try:
            records.append(
                _SECSubmissionRecord(
                    accession_number=str(accessions[index]),
                    form_type=str(form),
                    filing_date=date.fromisoformat(str(filing_dates[index])),
                    report_date=_date_or_none(report_dates[index]),
                    primary_document=primary_document,
                )
            )
        except ValueError as exc:
            raise FilingError(
                code=FilingErrorCode.MALFORMED_RESPONSE,
                message="SEC submissions payload contains an invalid filing record.",
                provider=SEC_FILING_PROVIDER,
            ) from exc
    return tuple(records)


def _column(payload: Mapping[str, Any], name: str) -> list[Any]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise FilingError(
            code=FilingErrorCode.MALFORMED_RESPONSE,
            message=f"SEC submissions recent.{name} must be a list.",
            provider=SEC_FILING_PROVIDER,
        )
    return value


def _company_from_payload(
    payload: Mapping[str, Any],
    requested_company: FilingCompany,
) -> FilingCompany:
    name = payload.get("name")
    return FilingCompany(
        cik=str(payload.get("cik", requested_company.cik)),
        company_id=requested_company.company_id,
        legal_name=(
            str(name) if isinstance(name, str) and name.strip() else requested_company.legal_name
        ),
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 404:
        raise FilingError(
            code=FilingErrorCode.NOT_FOUND,
            message="SEC filing source did not contain the requested resource.",
            provider=SEC_FILING_PROVIDER,
        )
    if response.status_code == 429:
        raise FilingError(
            code=FilingErrorCode.RATE_LIMITED,
            message="SEC filing source rate limited the request.",
            provider=SEC_FILING_PROVIDER,
            retryable=True,
        )
    if response.status_code >= 500:
        raise FilingError(
            code=FilingErrorCode.PROVIDER_UNAVAILABLE,
            message=f"SEC filing source returned HTTP {response.status_code}.",
            provider=SEC_FILING_PROVIDER,
            retryable=True,
        )
    if response.status_code >= 400:
        raise FilingError(
            code=FilingErrorCode.INVALID_REQUEST,
            message=f"SEC filing source returned HTTP {response.status_code}.",
            provider=SEC_FILING_PROVIDER,
        )


def _normalize_forms(forms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(_require_text("forms[]", form).upper() for form in forms))
    if not normalized:
        raise FilingError(
            code=FilingErrorCode.INVALID_REQUEST,
            message="forms must contain at least one form type",
            provider=SEC_FILING_PROVIDER,
        )
    return normalized


def _date_or_none(value: object) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value))


def _safe_file_name(value: str) -> str:
    file_name = Path(value).name.strip()
    if file_name == "" or file_name in {".", ".."}:
        raise FilingError(
            code=FilingErrorCode.MALFORMED_RESPONSE,
            message="SEC filing primary document name is invalid.",
            provider=SEC_FILING_PROVIDER,
        )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", file_name)


def _accession_no_dashes(value: str) -> str:
    return value.replace("-", "")


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
