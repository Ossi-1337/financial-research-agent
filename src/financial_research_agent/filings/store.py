from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Self

from financial_research_agent.documents import (
    DocumentExtractionMethod,
    DocumentExtractionStatus,
    DocumentRegion,
)
from financial_research_agent.filings.contracts import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingIngestionResult,
    FilingSource,
)
from financial_research_agent.settings import Settings

FILING_STORE_VERSION = 1


class FilingStore:
    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        stale_after: timedelta = timedelta(days=30),
    ) -> None:
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self.storage_path = storage_path
        self.stale_after = stale_after
        self._results: dict[str, FilingIngestionResult] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            storage_path=settings.local_paths.data_dir / "filings" / "filings_index.json",
            stale_after=timedelta(days=settings.data_sources.filing_cache_ttl_days),
        )

    def save_result(self, result: FilingIngestionResult) -> FilingIngestionResult:
        key = _result_key(result.source.provider, result.company.cik)
        with self._lock:
            self._results[key] = result
            self._save()
        return result

    def get_result(
        self,
        *,
        cik: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> FilingIngestionResult | None:
        normalized_cik = FilingCompany(cik=cik).cik
        with self._lock:
            candidates = [
                result
                for result in self._results.values()
                if result.company.cik == normalized_cik
                and (provider is None or result.source.provider == provider)
            ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda result: result.source.retrieved_at)
        return self._with_freshness(latest, now or datetime.now(UTC))

    def count(self) -> int:
        with self._lock:
            return len(self._results)

    def list(self) -> tuple[FilingIngestionResult, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._results.values(),
                    key=lambda result: result.source.retrieved_at,
                    reverse=True,
                )
            )

    def clear(self) -> int:
        with self._lock:
            deleted = len(self._results)
            self._results.clear()
            self._save()
            return deleted

    def _with_freshness(
        self,
        result: FilingIngestionResult,
        now: datetime,
    ) -> FilingIngestionResult:
        if result.source.retrieved_at + self.stale_after > now:
            return result
        warning = "Stored filing documents are stale; refresh before relying on them."
        source = replace(result.source, freshness_warning=warning)
        filings = tuple(
            replace(
                filing,
                source=replace(filing.source, freshness_warning=warning),
                warnings=tuple(dict.fromkeys((*filing.warnings, warning))),
            )
            for filing in result.filings
        )
        warnings = tuple(dict.fromkeys((*result.warnings, warning)))
        return replace(result, source=source, filings=filings, warnings=warnings)

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != FILING_STORE_VERSION:
                raise ValueError("unsupported filing store version")
            results_payload = payload.get("results", ())
            if not isinstance(results_payload, list):
                raise ValueError("filing results must be a list")
            self._results = {
                _result_key(result.source.provider, result.company.cik): result
                for result in (filing_ingestion_result_from_dict(item) for item in results_payload)
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"Could not load filing store: {self.storage_path}"
            raise ValueError(message) from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FILING_STORE_VERSION,
            "results": [result.to_dict() for result in self._results.values()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def filing_ingestion_result_from_dict(payload: Any) -> FilingIngestionResult:
    if not isinstance(payload, dict):
        raise ValueError("filing result must be an object")
    company = _company_from_payload(payload["company"])
    source = _source_from_payload(payload["source"])
    return FilingIngestionResult(
        company=company,
        filings=tuple(_filing_from_payload(item) for item in payload.get("filings", ())),
        chunks=tuple(_chunk_from_payload(item) for item in payload.get("chunks", ())),
        source=source,
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
    )


def _company_from_payload(payload: Any) -> FilingCompany:
    if not isinstance(payload, dict):
        raise ValueError("filing company must be an object")
    return FilingCompany(
        cik=str(payload["cik"]),
        company_id=_optional_payload_text(payload, "company_id"),
        legal_name=_optional_payload_text(payload, "legal_name"),
    )


def _source_from_payload(payload: Any) -> FilingSource:
    if not isinstance(payload, dict):
        raise ValueError("filing source must be an object")
    data_as_of = payload.get("data_as_of")
    return FilingSource(
        provider=str(payload["provider"]),
        provider_status=str(payload["provider_status"]),
        source_url=str(payload["source_url"]),
        retrieved_at=_datetime_from_payload(payload["retrieved_at"]),
        data_as_of=date.fromisoformat(data_as_of) if isinstance(data_as_of, str) else None,
        attribution=str(payload["attribution"]),
        freshness_warning=_optional_payload_text(payload, "freshness_warning"),
    )


def _filing_from_payload(payload: Any) -> FilingDocument:
    if not isinstance(payload, dict):
        raise ValueError("filing document must be an object")
    report_date = payload.get("report_date")
    publication_date = payload.get("publication_date")
    return FilingDocument(
        id=str(payload["id"]),
        company=_company_from_payload(payload["company"]),
        form_type=str(payload["form_type"]),
        accession_number=str(payload["accession_number"]),
        filing_date=date.fromisoformat(str(payload["filing_date"])),
        report_date=date.fromisoformat(report_date) if isinstance(report_date, str) else None,
        publication_date=(
            date.fromisoformat(publication_date) if isinstance(publication_date, str) else None
        ),
        document_url=str(payload["document_url"]),
        source_url=str(payload["source_url"]),
        document_format=FilingDocumentFormat(str(payload["document_format"])),
        retrieved_at=_datetime_from_payload(payload["retrieved_at"]),
        local_raw_path=str(payload["local_raw_path"]),
        local_text_path=str(payload["local_text_path"]),
        source=_source_from_payload(payload["source"]),
        chunk_ids=tuple(str(item) for item in payload.get("chunk_ids", ())),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        extraction_status=(
            DocumentExtractionStatus(str(payload["extraction_status"]))
            if payload.get("extraction_status") is not None
            else None
        ),
        extraction_method=(
            DocumentExtractionMethod(str(payload["extraction_method"]))
            if payload.get("extraction_method") is not None
            else None
        ),
        page_count=(int(payload["page_count"]) if payload.get("page_count") is not None else None),
        missing_text_pages=tuple(int(item) for item in payload.get("missing_text_pages", ())),
    )


def _chunk_from_payload(payload: Any) -> FilingChunk:
    if not isinstance(payload, dict):
        raise ValueError("filing chunk must be an object")
    return FilingChunk(
        id=str(payload["id"]),
        filing_id=str(payload["filing_id"]),
        chunk_index=int(payload["chunk_index"]),
        text=str(payload["text"]),
        char_start=int(payload["char_start"]),
        char_end=int(payload["char_end"]),
        section_heading=_optional_payload_text(payload, "section_heading"),
        source_url=str(payload["source_url"]),
        accession_number=str(payload["accession_number"]),
        form_type=str(payload["form_type"]),
        metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        source_region=(
            DocumentRegion.from_dict(payload["source_region"])
            if isinstance(payload.get("source_region"), dict)
            else None
        ),
        extraction_method=(
            DocumentExtractionMethod(str(payload["extraction_method"]))
            if payload.get("extraction_method") is not None
            else None
        ),
    )


def _datetime_from_payload(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _optional_payload_text(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    return str(value)


def _result_key(provider: str, cik: str) -> str:
    return f"{_require_text('provider', provider)}:{FilingCompany(cik=cik).cik}"


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
