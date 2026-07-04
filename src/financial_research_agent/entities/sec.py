from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from financial_research_agent.entities.contracts import (
    CompanySearchCandidate,
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
    SourceMetadata,
)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_PROVIDER = "sec_company_tickers"
SEC_PROVIDER_STATUS = "official"
SEC_ATTRIBUTION = "U.S. Securities and Exchange Commission company tickers"
SEC_CACHE_VERSION = 1
SEC_COVERAGE_WARNING = (
    "SEC company_tickers.json covers SEC filer ticker mappings and does not include exchange, "
    "currency, country, or ISIN. Confirm identifiers with OpenFIGI or exchange metadata later."
)


@dataclass(frozen=True, slots=True)
class SECCompanyTickerRecord:
    cik: int
    ticker: str
    title: str

    def __post_init__(self) -> None:
        if self.cik <= 0:
            raise ValueError("cik must be positive")
        object.__setattr__(self, "ticker", _require_text("ticker", self.ticker).upper())
        object.__setattr__(self, "title", _require_text("title", self.title))

    @property
    def padded_cik(self) -> str:
        return f"{self.cik:010d}"

    def to_dict(self) -> dict[str, object]:
        return {"cik": self.cik, "ticker": self.ticker, "title": self.title}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SECCompanyTickerRecord:
        try:
            return cls(
                cik=int(payload["cik"]),
                ticker=str(payload["ticker"]),
                title=str(payload["title"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid cached SEC ticker record") from exc


class SECCompanyTickerProvider:
    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        cache_ttl: timedelta = timedelta(days=30),
        user_agent: str = "financial-research-agent/0.1 local-research contact-not-configured",
        http_client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if cache_ttl <= timedelta(0):
            raise ValueError("cache_ttl must be positive")
        self.cache_path = cache_path
        self.cache_ttl = cache_ttl
        self.user_agent = _require_text("user_agent", user_agent)
        self._http_client = http_client
        self._now = now or (lambda: datetime.now(UTC))

    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        text = _require_text("query", query)
        if limit <= 0 or limit > 50:
            raise CompanySearchError(
                code=CompanySearchErrorCode.INVALID_REQUEST,
                message="limit must be between 1 and 50",
                provider=SEC_PROVIDER,
            )
        records, source, warnings = await self._records()
        scored = [
            (score, reason, record)
            for record in records
            for score, reason in [_score_record(text, record)]
            if score > 0
        ]
        scored.sort(key=lambda item: (-item[0], item[2].title, item[2].ticker))
        candidates = tuple(
            _candidate_from_record(record, score=score, match_reason=reason, source=source)
            for score, reason, record in scored[:limit]
        )
        status = (
            CompanySearchStatus.REVIEW_REQUIRED if candidates else CompanySearchStatus.NO_MATCHES
        )
        result_warnings = warnings
        if not candidates:
            result_warnings = (
                *result_warnings,
                "No SEC company ticker matches found. Coverage is limited to SEC filers.",
            )
        return CompanySearchResult(
            query=text,
            status=status,
            candidates=candidates,
            source=source,
            warnings=result_warnings,
        )

    async def _records(
        self,
    ) -> tuple[tuple[SECCompanyTickerRecord, ...], SourceMetadata, tuple[str, ...]]:
        cached = self._read_cache(require_fresh=True)
        if cached is not None:
            return cached
        try:
            fetched = await self._fetch_records()
        except CompanySearchError:
            stale = self._read_cache(require_fresh=False)
            if stale is None:
                raise
            records, source, warnings = stale
            return (
                records,
                SourceMetadata(
                    provider=source.provider,
                    provider_status=source.provider_status,
                    source_url=source.source_url,
                    retrieved_at=source.retrieved_at,
                    cache_expires_at=source.cache_expires_at,
                    data_as_of=source.data_as_of,
                    attribution=source.attribution,
                    freshness_warning="Using stale cached SEC company ticker data.",
                ),
                (*warnings, "Live SEC refresh failed; stale cached data was used."),
            )
        self._write_cache(*fetched)
        return fetched

    async def _fetch_records(
        self,
    ) -> tuple[tuple[SECCompanyTickerRecord, ...], SourceMetadata, tuple[str, ...]]:
        now = self._now()
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        try:
            if self._http_client is None:
                async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                    response = await client.get(SEC_COMPANY_TICKERS_URL)
            else:
                response = await self._http_client.get(SEC_COMPANY_TICKERS_URL, headers=headers)
        except httpx.TimeoutException as exc:
            raise CompanySearchError(
                code=CompanySearchErrorCode.TIMEOUT,
                message="Timed out while fetching SEC company tickers.",
                provider=SEC_PROVIDER,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise CompanySearchError(
                code=CompanySearchErrorCode.PROVIDER_UNAVAILABLE,
                message=f"SEC company ticker source is unavailable: {exc}",
                provider=SEC_PROVIDER,
                retryable=True,
            ) from exc
        if response.status_code == 429:
            raise CompanySearchError(
                code=CompanySearchErrorCode.RATE_LIMITED,
                message="SEC company ticker source rate limited the request.",
                provider=SEC_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 500:
            raise CompanySearchError(
                code=CompanySearchErrorCode.PROVIDER_UNAVAILABLE,
                message=f"SEC company ticker source returned HTTP {response.status_code}.",
                provider=SEC_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 400:
            raise CompanySearchError(
                code=CompanySearchErrorCode.INVALID_REQUEST,
                message=f"SEC company ticker source returned HTTP {response.status_code}.",
                provider=SEC_PROVIDER,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CompanySearchError(
                code=CompanySearchErrorCode.MALFORMED_RESPONSE,
                message="SEC company ticker source returned malformed JSON.",
                provider=SEC_PROVIDER,
            ) from exc
        records = _parse_sec_payload(payload)
        return records, _source_metadata(now, self.cache_ttl), (SEC_COVERAGE_WARNING,)

    def _read_cache(
        self,
        *,
        require_fresh: bool,
    ) -> tuple[tuple[SECCompanyTickerRecord, ...], SourceMetadata, tuple[str, ...]] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != SEC_CACHE_VERSION:
                return None
            retrieved_at = datetime.fromisoformat(str(payload["retrieved_at"]))
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=UTC)
            cache_expires_at = retrieved_at + self.cache_ttl
            if require_fresh and cache_expires_at <= self._now():
                return None
            records_payload = payload["records"]
            if not isinstance(records_payload, list):
                return None
            records = tuple(SECCompanyTickerRecord.from_dict(item) for item in records_payload)
        except OSError, ValueError, TypeError, KeyError, json.JSONDecodeError:
            return None
        source = SourceMetadata(
            provider=SEC_PROVIDER,
            provider_status=SEC_PROVIDER_STATUS,
            source_url=SEC_COMPANY_TICKERS_URL,
            retrieved_at=retrieved_at,
            cache_expires_at=cache_expires_at,
            attribution=SEC_ATTRIBUTION,
            freshness_warning=(
                "Using stale cached SEC company ticker data."
                if cache_expires_at <= self._now()
                else None
            ),
        )
        return records, source, (SEC_COVERAGE_WARNING,)

    def _write_cache(
        self,
        records: tuple[SECCompanyTickerRecord, ...],
        source: SourceMetadata,
        _warnings: tuple[str, ...],
    ) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SEC_CACHE_VERSION,
            "retrieved_at": source.retrieved_at.isoformat(),
            "source_url": source.source_url,
            "records": [record.to_dict() for record in records],
        }
        temp_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.cache_path)


def _parse_sec_payload(payload: Any) -> tuple[SECCompanyTickerRecord, ...]:
    if not isinstance(payload, Mapping):
        raise CompanySearchError(
            code=CompanySearchErrorCode.MALFORMED_RESPONSE,
            message="SEC company ticker payload must be an object.",
            provider=SEC_PROVIDER,
        )
    records: list[SECCompanyTickerRecord] = []
    for value in payload.values():
        if not isinstance(value, Mapping):
            raise CompanySearchError(
                code=CompanySearchErrorCode.MALFORMED_RESPONSE,
                message="SEC company ticker payload contains a non-object record.",
                provider=SEC_PROVIDER,
            )
        try:
            records.append(
                SECCompanyTickerRecord(
                    cik=int(value["cik_str"]),
                    ticker=str(value["ticker"]),
                    title=str(value["title"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CompanySearchError(
                code=CompanySearchErrorCode.MALFORMED_RESPONSE,
                message="SEC company ticker payload contains an invalid record.",
                provider=SEC_PROVIDER,
            ) from exc
    return tuple(records)


def _candidate_from_record(
    record: SECCompanyTickerRecord,
    *,
    score: float,
    match_reason: str,
    source: SourceMetadata,
) -> CompanySearchCandidate:
    cik_identifier = EntityIdentifier(
        identifier_type=EntityIdentifierType.CIK,
        value=record.padded_cik,
        source=SEC_PROVIDER,
    )
    ticker_identifier = EntityIdentifier(
        identifier_type=EntityIdentifierType.TICKER,
        value=record.ticker,
        source=SEC_PROVIDER,
    )
    company_id = f"sec:cik:{record.padded_cik}"
    company = ResolvedCompany(
        id=company_id,
        legal_name=record.title,
        display_name=record.title,
        identifiers=(cik_identifier, ticker_identifier),
    )
    security = ResolvedSecurity(
        id=f"sec:ticker:{record.ticker}:cik:{record.padded_cik}",
        company_id=company_id,
        ticker=record.ticker,
        name=record.title,
        identifiers=(ticker_identifier, cik_identifier),
    )
    return CompanySearchCandidate(
        company=company,
        securities=(security,),
        score=score,
        match_reason=match_reason,
        source=source,
        warnings=(SEC_COVERAGE_WARNING,),
    )


def _score_record(query: str, record: SECCompanyTickerRecord) -> tuple[float, str]:
    ticker_query = query.strip().upper()
    normalized_query = _normalize(query)
    normalized_title = _normalize(record.title)
    if ticker_query == record.ticker:
        return 100.0, "ticker_exact"
    if normalized_query == normalized_title:
        return 96.0, "company_name_exact"
    if normalized_title.startswith(normalized_query):
        return 90.0, "company_name_prefix"
    query_tokens = normalized_query.split()
    title_tokens = normalized_title.split()
    if query_tokens and all(token in title_tokens for token in query_tokens):
        return 84.0, "company_name_tokens"
    if normalized_query and normalized_query in normalized_title:
        return 72.0, "company_name_contains"
    if ticker_query and record.ticker.startswith(ticker_query):
        return 60.0, "ticker_prefix"
    return 0.0, "no_match"


def _normalize(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(text.split())


def _source_metadata(retrieved_at: datetime, cache_ttl: timedelta) -> SourceMetadata:
    return SourceMetadata(
        provider=SEC_PROVIDER,
        provider_status=SEC_PROVIDER_STATUS,
        source_url=SEC_COMPANY_TICKERS_URL,
        retrieved_at=retrieved_at,
        cache_expires_at=retrieved_at + cache_ttl,
        attribution=SEC_ATTRIBUTION,
    )


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
