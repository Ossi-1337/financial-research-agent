from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType


class ContextAnalysisStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_RELIABLE_SOURCES = "no_reliable_sources"


class ContextScope(StrEnum):
    COMPANY = "company"
    MACRO = "macro"
    SECTOR = "sector"


class ContextSourceType(StrEnum):
    COMPANY_NEWS = "company_news"
    COMPANY_EVENT = "company_event"
    MACRO_INDICATOR = "macro_indicator"
    RATES = "rates"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    SECTOR_CONTEXT = "sector_context"


class SourceReliability(StrEnum):
    OFFICIAL = "official"
    REGULATORY = "regulatory"
    DOCUMENTED_API = "documented_api"
    COMPANY_SOURCE = "company_source"
    REPUTABLE_NEWS = "reputable_news"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class ContextRecency(StrEnum):
    RECENT = "recent"
    STALE = "stale"
    UNDATED = "undated"
    FUTURE_DATED = "future_dated"


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ContextSourceItem:
    id: str
    title: str
    summary: str
    source_url: str
    source_name: str
    source_type: ContextSourceType
    reliability: SourceReliability
    scope: ContextScope
    retrieved_at: datetime
    published_at: datetime | None = None
    company_symbols: tuple[str, ...] = ()
    sector: str | None = None
    region: str | None = None
    topics: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "source_url", _require_http_url("source_url", self.source_url))
        object.__setattr__(self, "source_name", _require_text("source_name", self.source_name))
        object.__setattr__(self, "source_type", ContextSourceType(self.source_type))
        object.__setattr__(self, "reliability", SourceReliability(self.reliability))
        object.__setattr__(self, "scope", ContextScope(self.scope))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        if self.published_at is not None:
            object.__setattr__(
                self,
                "published_at",
                _aware_datetime("published_at", self.published_at),
            )
        object.__setattr__(
            self,
            "company_symbols",
            _upper_text_tuple("company_symbols", self.company_symbols),
        )
        object.__setattr__(self, "sector", _optional_text(self.sector))
        object.__setattr__(self, "region", _optional_upper_text(self.region))
        object.__setattr__(self, "topics", _text_tuple("topics", self.topics))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "reliability": self.reliability.value,
            "scope": self.scope.value,
            "retrieved_at": self.retrieved_at.isoformat(),
            "published_at": (
                self.published_at.isoformat() if self.published_at is not None else None
            ),
            "company_symbols": list(self.company_symbols),
            "sector": self.sector,
            "region": self.region,
            "topics": list(self.topics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ContextFinding:
    id: str
    scope: ContextScope
    title: str
    summary: str
    confidence: ConfidenceLabel
    source_item_ids: tuple[str, ...] = ()
    recency: ContextRecency = ContextRecency.RECENT
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "scope", ContextScope(self.scope))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "confidence", ConfidenceLabel(self.confidence))
        object.__setattr__(
            self,
            "source_item_ids",
            _text_tuple("source_item_ids", self.source_item_ids),
        )
        object.__setattr__(self, "recency", ContextRecency(self.recency))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        if not self.source_item_ids and not self.limitations:
            raise ValueError("finding must include source_item_ids or limitations")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "title": self.title,
            "summary": self.summary,
            "confidence": self.confidence.value,
            "source_item_ids": list(self.source_item_ids),
            "recency": self.recency.value,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ContextSourceStrategyItem:
    category: ContextSourceType
    primary_sources: tuple[str, ...]
    fallback_sources: tuple[str, ...] = ()
    reliability_notes: str = ""
    freshness_guidance: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", ContextSourceType(self.category))
        object.__setattr__(
            self,
            "primary_sources",
            _text_tuple("primary_sources", self.primary_sources),
        )
        object.__setattr__(
            self,
            "fallback_sources",
            _text_tuple("fallback_sources", self.fallback_sources),
        )
        object.__setattr__(
            self,
            "reliability_notes",
            _require_text("reliability_notes", self.reliability_notes),
        )
        object.__setattr__(
            self,
            "freshness_guidance",
            _require_text("freshness_guidance", self.freshness_guidance),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "primary_sources": list(self.primary_sources),
            "fallback_sources": list(self.fallback_sources),
            "reliability_notes": self.reliability_notes,
            "freshness_guidance": self.freshness_guidance,
        }


@dataclass(frozen=True, slots=True)
class ContextAnalysisResult:
    id: str
    query: str
    status: ContextAnalysisStatus
    created_at: datetime
    source_items: tuple[ContextSourceItem, ...]
    findings: tuple[ContextFinding, ...]
    source_strategy: tuple[ContextSourceStrategyItem, ...]
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", ContextAnalysisStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "source_items", _source_item_tuple(self.source_items))
        object.__setattr__(self, "findings", _finding_tuple(self.findings))
        object.__setattr__(self, "source_strategy", _strategy_tuple(self.source_strategy))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        _validate_result_references(self.source_items, self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "source_items": [item.to_dict() for item in self.source_items],
            "findings": [finding.to_dict() for finding in self.findings],
            "source_strategy": [item.to_dict() for item in self.source_strategy],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _require_http_url(name: str, value: str) -> str:
    text = _require_text(name, value)
    if not text.startswith(("https://", "http://")):
        raise ValueError(f"{name} must be an http(s) URL")
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


def _upper_text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value.upper() for value in _text_tuple(name, values))


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _source_item_tuple(values: Iterable[ContextSourceItem]) -> tuple[ContextSourceItem, ...]:
    items = tuple(values)
    for index, item in enumerate(items):
        if not isinstance(item, ContextSourceItem):
            raise ValueError(f"source_items[{index}] must be a ContextSourceItem")
    return items


def _finding_tuple(values: Iterable[ContextFinding]) -> tuple[ContextFinding, ...]:
    findings = tuple(values)
    for index, finding in enumerate(findings):
        if not isinstance(finding, ContextFinding):
            raise ValueError(f"findings[{index}] must be a ContextFinding")
    return findings


def _strategy_tuple(
    values: Iterable[ContextSourceStrategyItem],
) -> tuple[ContextSourceStrategyItem, ...]:
    items = tuple(values)
    for index, item in enumerate(items):
        if not isinstance(item, ContextSourceStrategyItem):
            raise ValueError(f"source_strategy[{index}] must be a ContextSourceStrategyItem")
    return items


def _validate_result_references(
    source_items: tuple[ContextSourceItem, ...],
    findings: tuple[ContextFinding, ...],
) -> None:
    source_ids = [item.id for item in source_items]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_items must have unique ids")
    known_source_ids = set(source_ids)
    for finding in findings:
        for source_item_id in finding.source_item_ids:
            if source_item_id not in known_source_ids:
                raise ValueError(
                    f"finding {finding.id!r} references unknown source item {source_item_id!r}"
                )
