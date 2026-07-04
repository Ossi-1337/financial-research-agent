from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class EntityIdentifierType(StrEnum):
    CIK = "cik"
    TICKER = "ticker"
    ISIN = "isin"
    LEI = "lei"
    FIGI = "figi"


class CompanySearchStatus(StrEnum):
    NO_MATCHES = "no_matches"
    REVIEW_REQUIRED = "review_required"


class CompanySearchErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"


@dataclass(frozen=True, slots=True)
class EntityIdentifier:
    identifier_type: EntityIdentifierType
    value: str
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier_type", EntityIdentifierType(self.identifier_type))
        object.__setattr__(self, "value", _require_text("value", self.value))
        object.__setattr__(self, "source", _optional_text(self.source))

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.identifier_type.value,
            "value": self.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    provider: str
    provider_status: str
    source_url: str
    retrieved_at: datetime
    attribution: str
    cache_expires_at: datetime | None = None
    data_as_of: datetime | None = None
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
        if self.cache_expires_at is not None:
            object.__setattr__(
                self,
                "cache_expires_at",
                _aware_datetime("cache_expires_at", self.cache_expires_at),
            )
        if self.data_as_of is not None:
            object.__setattr__(self, "data_as_of", _aware_datetime("data_as_of", self.data_as_of))
        object.__setattr__(self, "freshness_warning", _optional_text(self.freshness_warning))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_status": self.provider_status,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "cache_expires_at": (
                self.cache_expires_at.isoformat() if self.cache_expires_at is not None else None
            ),
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of is not None else None,
            "attribution": self.attribution,
            "freshness_warning": self.freshness_warning,
        }


@dataclass(frozen=True, slots=True)
class ResolvedCompany:
    id: str
    legal_name: str
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    identifiers: tuple[EntityIdentifier, ...] = ()
    country_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "legal_name", _require_text("legal_name", self.legal_name))
        object.__setattr__(self, "display_name", _optional_text(self.display_name))
        object.__setattr__(self, "aliases", _text_tuple("aliases", self.aliases))
        object.__setattr__(self, "identifiers", _identifier_tuple(self.identifiers))
        object.__setattr__(self, "country_code", _optional_upper_text(self.country_code))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "legal_name": self.legal_name,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "identifiers": [identifier.to_dict() for identifier in self.identifiers],
            "country_code": self.country_code,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSecurity:
    id: str
    company_id: str
    ticker: str
    name: str
    exchange_mic: str | None = None
    exchange_name: str | None = None
    currency: str | None = None
    country_code: str | None = None
    isin: str | None = None
    identifiers: tuple[EntityIdentifier, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "company_id", _require_text("company_id", self.company_id))
        object.__setattr__(self, "ticker", _require_text("ticker", self.ticker).upper())
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "exchange_mic", _optional_upper_text(self.exchange_mic))
        object.__setattr__(self, "exchange_name", _optional_text(self.exchange_name))
        object.__setattr__(self, "currency", _optional_upper_text(self.currency))
        object.__setattr__(self, "country_code", _optional_upper_text(self.country_code))
        object.__setattr__(self, "isin", _optional_upper_text(self.isin))
        object.__setattr__(self, "identifiers", _identifier_tuple(self.identifiers))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "ticker": self.ticker,
            "name": self.name,
            "exchange_mic": self.exchange_mic,
            "exchange_name": self.exchange_name,
            "currency": self.currency,
            "country_code": self.country_code,
            "isin": self.isin,
            "identifiers": [identifier.to_dict() for identifier in self.identifiers],
        }


@dataclass(frozen=True, slots=True)
class CompanySearchCandidate:
    company: ResolvedCompany
    securities: tuple[ResolvedSecurity, ...]
    score: float
    match_reason: str
    source: SourceMetadata
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.company, ResolvedCompany):
            raise ValueError("company must be a ResolvedCompany")
        securities = tuple(self.securities)
        if not securities:
            raise ValueError("securities must contain at least one security")
        for index, security in enumerate(securities):
            if not isinstance(security, ResolvedSecurity):
                raise ValueError(f"securities[{index}] must be a ResolvedSecurity")
        if self.score < 0:
            raise ValueError("score must be non-negative")
        if not isinstance(self.source, SourceMetadata):
            raise ValueError("source must be SourceMetadata")
        object.__setattr__(self, "securities", securities)
        object.__setattr__(self, "match_reason", _require_text("match_reason", self.match_reason))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "company": self.company.to_dict(),
            "securities": [security.to_dict() for security in self.securities],
            "score": self.score,
            "match_reason": self.match_reason,
            "source": self.source.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class CompanySearchResult:
    query: str
    status: CompanySearchStatus
    candidates: tuple[CompanySearchCandidate, ...] = ()
    source: SourceMetadata | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", CompanySearchStatus(self.status))
        candidates = tuple(self.candidates)
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, CompanySearchCandidate):
                raise ValueError(f"candidates[{index}] must be a CompanySearchCandidate")
        if self.source is not None and not isinstance(self.source, SourceMetadata):
            raise ValueError("source must be SourceMetadata")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "status": self.status.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "source": self.source.to_dict() if self.source is not None else None,
            "warnings": list(self.warnings),
        }


class CompanySearchProvider(Protocol):
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult: ...


class CompanySearchError(Exception):
    def __init__(
        self,
        *,
        code: CompanySearchErrorCode | str,
        message: str,
        provider: str,
        retryable: bool = False,
    ) -> None:
        self.code = CompanySearchErrorCode(code)
        self.message = _require_text("message", message)
        self.provider = _require_text("provider", provider)
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "company_search_error",
            "code": self.code.value,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
        }


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


def _optional_upper_text(value: str | None) -> str | None:
    text = _optional_text(value)
    return text.upper() if text is not None else None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _identifier_tuple(values: Iterable[EntityIdentifier]) -> tuple[EntityIdentifier, ...]:
    identifiers = tuple(values)
    for index, identifier in enumerate(identifiers):
        if not isinstance(identifier, EntityIdentifier):
            raise ValueError(f"identifiers[{index}] must be an EntityIdentifier")
    return identifiers


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
