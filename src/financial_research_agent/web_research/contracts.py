from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class WebResearchStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_SOURCES = "no_sources"
    UNAVAILABLE = "unavailable"


class WebJurisdiction(StrEnum):
    DK = "DK"
    EU = "EU"
    US = "US"


class WebSourceType(StrEnum):
    NEWS = "news"
    REGULATORY = "regulatory"
    COMPANY = "company"
    MACRO = "macro"
    SECONDARY = "secondary"


class WebSourceReliability(StrEnum):
    OFFICIAL = "official"
    REGULATORY = "regulatory"
    DOCUMENTED_API = "documented_api"
    COMPANY_SOURCE = "company_source"
    REPUTABLE_NEWS = "reputable_news"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class WebResearchErrorCode(StrEnum):
    DISABLED = "web_research_disabled"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    UNSAFE_URL = "unsafe_url"
    UNSUPPORTED_CONTENT = "unsupported_content"
    CONTENT_TOO_LARGE = "content_too_large"
    FETCH_FAILED = "fetch_failed"


@dataclass(frozen=True, slots=True)
class WebSearchCandidate:
    title: str
    url: str
    snippet: str
    provider: str
    source_name: str
    published_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "url", _require_https_url("url", self.url))
        object.__setattr__(self, "snippet", _require_text("snippet", self.snippet))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "source_name", _require_text("source_name", self.source_name))
        if self.published_at is not None:
            object.__setattr__(
                self,
                "published_at",
                _aware_datetime("published_at", self.published_at),
            )
        object.__setattr__(self, "metadata", MappingProxyType(_text_mapping(self.metadata)))


@dataclass(frozen=True, slots=True)
class WebSourceEvidence:
    id: str
    canonical_url: str
    title: str
    publisher: str
    quote: str
    source_type: WebSourceType
    reliability: WebSourceReliability
    provider: str
    retrieved_at: datetime
    expires_at: datetime
    content_sha256: str
    jurisdiction: WebJurisdiction | None = None
    published_at: datetime | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(
            self,
            "canonical_url",
            _require_https_url("canonical_url", self.canonical_url),
        )
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "publisher", _require_text("publisher", self.publisher))
        quote = _require_text("quote", self.quote)
        if len(quote) > 1_500:
            raise ValueError("quote must contain at most 1500 characters")
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "source_type", WebSourceType(self.source_type))
        object.__setattr__(self, "reliability", WebSourceReliability(self.reliability))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(self, "expires_at", _aware_datetime("expires_at", self.expires_at))
        if self.expires_at <= self.retrieved_at:
            raise ValueError("expires_at must be after retrieved_at")
        object.__setattr__(self, "content_sha256", _require_sha256(self.content_sha256))
        if self.jurisdiction is not None:
            object.__setattr__(self, "jurisdiction", WebJurisdiction(self.jurisdiction))
        if self.published_at is not None:
            object.__setattr__(
                self,
                "published_at",
                _aware_datetime("published_at", self.published_at),
            )
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "metadata", MappingProxyType(_text_mapping(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "publisher": self.publisher,
            "quote": self.quote,
            "source_type": self.source_type.value,
            "reliability": self.reliability.value,
            "provider": self.provider,
            "jurisdiction": self.jurisdiction.value if self.jurisdiction else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "content_sha256": self.content_sha256,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> WebSourceEvidence:
        published_at = value.get("published_at")
        jurisdiction = value.get("jurisdiction")
        return cls(
            id=str(value["id"]),
            canonical_url=str(value["canonical_url"]),
            title=str(value["title"]),
            publisher=str(value["publisher"]),
            quote=str(value["quote"]),
            source_type=str(value["source_type"]),
            reliability=str(value["reliability"]),
            provider=str(value["provider"]),
            jurisdiction=str(jurisdiction) if jurisdiction else None,
            published_at=datetime.fromisoformat(str(published_at)) if published_at else None,
            retrieved_at=datetime.fromisoformat(str(value["retrieved_at"])),
            expires_at=datetime.fromisoformat(str(value["expires_at"])),
            content_sha256=str(value["content_sha256"]),
            warnings=tuple(str(item) for item in value.get("warnings", ())),
            metadata={str(key): str(item) for key, item in _mapping(value.get("metadata")).items()},
        )


@dataclass(frozen=True, slots=True)
class WebResearchRequest:
    query: str
    jurisdiction: WebJurisdiction | None = None
    company_name: str | None = None
    ticker: str | None = None
    requires_official_source: bool = False

    def __post_init__(self) -> None:
        query = _require_text("query", self.query)
        if len(query) > 500:
            raise ValueError("query must contain at most 500 characters")
        object.__setattr__(self, "query", query)
        if self.jurisdiction is not None:
            object.__setattr__(self, "jurisdiction", WebJurisdiction(self.jurisdiction))
        object.__setattr__(self, "company_name", _optional_text(self.company_name))
        ticker = _optional_text(self.ticker)
        object.__setattr__(self, "ticker", ticker.upper() if ticker else None)


@dataclass(frozen=True, slots=True)
class WebResearchResult:
    status: WebResearchStatus
    sources: tuple[WebSourceEvidence, ...]
    warnings: tuple[str, ...] = ()
    no_result_reason: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", WebResearchStatus(self.status))
        sources = tuple(self.sources)
        if not all(isinstance(source, WebSourceEvidence) for source in sources):
            raise ValueError("sources must contain WebSourceEvidence values")
        if len(sources) > 5:
            raise ValueError("sources must contain at most five values")
        if sum(len(source.quote) for source in sources) > 5_000:
            raise ValueError("source quotes must contain at most 5000 characters")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "no_result_reason", _optional_text(self.no_result_reason))
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "sources": [source.to_dict() for source in self.sources],
            "warnings": list(self.warnings),
            "no_result_reason": self.no_result_reason,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class CitedContextAnswer:
    answer: str
    evidence_ids: tuple[str, ...]
    jurisdiction: WebJurisdiction | None = None
    source_dates: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    no_legal_advice_notice: str = (
        "This is source-backed research and education, not legal or accounting advice."
    )

    def __post_init__(self) -> None:
        answer = _require_text("answer", self.answer)
        if len(answer) > 4_000:
            raise ValueError("answer must contain at most 4000 characters")
        object.__setattr__(self, "answer", answer)
        object.__setattr__(
            self,
            "evidence_ids",
            _text_tuple("evidence_ids", self.evidence_ids),
        )
        if self.jurisdiction is not None:
            object.__setattr__(self, "jurisdiction", WebJurisdiction(self.jurisdiction))
        object.__setattr__(
            self,
            "source_dates",
            MappingProxyType(_text_mapping(self.source_dates)),
        )
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(
            self,
            "limitations",
            _text_tuple("limitations", self.limitations),
        )
        object.__setattr__(
            self,
            "no_legal_advice_notice",
            _require_text("no_legal_advice_notice", self.no_legal_advice_notice),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
            "jurisdiction": self.jurisdiction.value if self.jurisdiction else None,
            "source_dates": dict(self.source_dates),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "no_legal_advice_notice": self.no_legal_advice_notice,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CitedContextAnswer:
        jurisdiction = value.get("jurisdiction")
        return cls(
            answer=str(value["answer"]),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", ())),
            jurisdiction=str(jurisdiction) if jurisdiction else None,
            source_dates={
                str(key): str(item) for key, item in _mapping(value.get("source_dates")).items()
            },
            warnings=tuple(str(item) for item in value.get("warnings", ())),
            limitations=tuple(str(item) for item in value.get("limitations", ())),
            no_legal_advice_notice=str(
                value.get(
                    "no_legal_advice_notice",
                    "This is source-backed research and education, not legal or accounting advice.",
                )
            ),
        )


class WebSearchProvider(Protocol):
    async def search(
        self,
        request: WebResearchRequest,
        *,
        limit: int,
    ) -> tuple[WebSearchCandidate, ...]: ...


class WebSourceCache(Protocol):
    def get(self, canonical_url: str, *, now: datetime) -> WebSourceEvidence | None: ...

    def save(self, source: WebSourceEvidence) -> WebSourceEvidence: ...


class WebResearchError(Exception):
    def __init__(self, code: WebResearchErrorCode | str, message: str) -> None:
        self.code = WebResearchErrorCode(code)
        self.message = _require_text("message", message)
        super().__init__(self.message)


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_https_url(name: str, value: str) -> str:
    text = _require_text(name, value)
    if not text.lower().startswith("https://"):
        raise ValueError(f"{name} must use HTTPS")
    return text


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_sha256(value: str) -> str:
    text = _require_text("content_sha256", value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("content_sha256 must be a hexadecimal SHA-256 digest")
    return text


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _text_mapping(values: Mapping[str, str]) -> dict[str, str]:
    return {
        _require_text("metadata key", str(key)): _require_text("metadata value", str(value))
        for key, value in values.items()
    }


def _mapping(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}
