from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from financial_research_agent.context_analysis.contracts import (
    ConfidenceLabel,
    ContextAnalysisResult,
    ContextAnalysisStatus,
    ContextFinding,
    ContextRecency,
    ContextScope,
    ContextSourceItem,
    ContextSourceStrategyItem,
    ContextSourceType,
    SourceReliability,
)

DEFAULT_RECENT_WINDOW_DAYS = 14
RELIABLE_SOURCE_TYPES = {
    SourceReliability.OFFICIAL,
    SourceReliability.REGULATORY,
    SourceReliability.DOCUMENTED_API,
    SourceReliability.COMPANY_SOURCE,
    SourceReliability.REPUTABLE_NEWS,
}


class NewsMacroSectorAgent:
    """Deterministic context agent over explicit source-linked context items."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    ) -> None:
        if recent_window_days <= 0:
            raise ValueError("recent_window_days must be positive")
        self._now = now or (lambda: datetime.now(UTC))
        self._recent_window = timedelta(days=recent_window_days)

    def analyze(
        self,
        *,
        query: str,
        source_items: Iterable[ContextSourceItem],
        company_symbols: tuple[str, ...] = (),
        sector: str | None = None,
        region: str | None = None,
    ) -> ContextAnalysisResult:
        created_at = _aware_now(self._now())
        normalized_symbols = tuple(
            symbol.strip().upper() for symbol in company_symbols if symbol.strip()
        )
        normalized_sector = _optional_text(sector)
        normalized_region = _optional_upper_text(region)
        items = _deduplicate_sources(tuple(source_items))
        warnings = _dedupe_warnings(source_items=tuple(source_items), deduped_items=items)
        findings = (
            self._finding_for_scope(
                scope=ContextScope.COMPANY,
                title="Company-Specific Events",
                items=_scope_items(
                    items,
                    ContextScope.COMPANY,
                    company_symbols=normalized_symbols,
                ),
                created_at=created_at,
                required=True,
            ),
            self._finding_for_scope(
                scope=ContextScope.MACRO,
                title="Macro Context",
                items=_scope_items(items, ContextScope.MACRO, region=normalized_region),
                created_at=created_at,
                required=True,
            ),
            self._finding_for_scope(
                scope=ContextScope.SECTOR,
                title="Sector Context",
                items=_scope_items(items, ContextScope.SECTOR, sector=normalized_sector),
                created_at=created_at,
                required=True,
            ),
        )
        top_level_limitations = tuple(
            dict.fromkeys(limitation for finding in findings for limitation in finding.limitations)
        )
        status = _status(findings, top_level_limitations)
        return ContextAnalysisResult(
            id=f"context_analysis_{uuid4().hex}",
            query=_require_text("query", query),
            status=status,
            created_at=created_at,
            source_items=items,
            findings=findings,
            source_strategy=create_default_context_source_strategy(),
            warnings=warnings,
            limitations=top_level_limitations,
        )

    def _finding_for_scope(
        self,
        *,
        scope: ContextScope,
        title: str,
        items: tuple[ContextSourceItem, ...],
        created_at: datetime,
        required: bool,
    ) -> ContextFinding:
        reliable = tuple(item for item in items if item.reliability in RELIABLE_SOURCE_TYPES)
        recent = tuple(
            item
            for item in reliable
            if _recency(item, created_at, self._recent_window) == ContextRecency.RECENT
        )
        source_pool = recent or reliable
        if not source_pool:
            limitation = (
                f"No reliable recent {scope.value.replace('_', ' ')} source was available."
                if required
                else f"No {scope.value.replace('_', ' ')} context was requested."
            )
            return ContextFinding(
                id=f"finding:{scope.value}",
                scope=scope,
                title=title,
                summary=limitation,
                confidence=ConfidenceLabel.UNKNOWN,
                recency=ContextRecency.UNDATED,
                limitations=(limitation,),
            )

        latest = max(source_pool, key=lambda item: item.published_at or item.retrieved_at)
        recency = _combined_recency(source_pool, created_at, self._recent_window)
        warnings = _scope_warnings(source_pool, recency)
        return ContextFinding(
            id=f"finding:{scope.value}",
            scope=scope,
            title=title,
            summary=_summary(scope, source_pool, latest),
            confidence=_confidence(source_pool, recency),
            source_item_ids=tuple(item.id for item in source_pool[:5]),
            recency=recency,
            warnings=warnings,
            limitations=(
                ()
                if recent
                else (f"{title} uses non-recent source items; refresh before relying on it.",)
            ),
        )


def create_default_context_source_strategy() -> tuple[ContextSourceStrategyItem, ...]:
    return (
        ContextSourceStrategyItem(
            category=ContextSourceType.COMPANY_NEWS,
            primary_sources=(
                "Company investor relations",
                "Regulatory filings",
                "Reputable news APIs",
            ),
            fallback_sources=("Documented commercial news APIs",),
            reliability_notes=(
                "Prefer company/regulatory primary sources; use accessible news links only."
            ),
            freshness_guidance="Recent company events should usually be within 14 days.",
        ),
        ContextSourceStrategyItem(
            category=ContextSourceType.MACRO_INDICATOR,
            primary_sources=("FRED", "ECB Data Portal", "Official statistics agencies"),
            fallback_sources=("Documented macro data APIs",),
            reliability_notes="Prefer official macro time series with release dates.",
            freshness_guidance="Use indicator-specific release calendars and data-as-of dates.",
        ),
        ContextSourceStrategyItem(
            category=ContextSourceType.RATES,
            primary_sources=("Central banks", "Treasury departments", "FRED"),
            fallback_sources=("Documented market-data APIs",),
            reliability_notes="Rates need source timestamp and series identifier.",
            freshness_guidance="Flag delayed or stale rates when outside provider TTL.",
        ),
        ContextSourceStrategyItem(
            category=ContextSourceType.CURRENCY,
            primary_sources=("ECB Data Portal", "Central banks", "Documented FX APIs"),
            fallback_sources=("Market data providers with clear terms",),
            reliability_notes="FX data must identify base/quote currency and timestamp.",
            freshness_guidance=(
                "Flag whether values are daily fixing, delayed, or latest available."
            ),
        ),
        ContextSourceStrategyItem(
            category=ContextSourceType.COMMODITY,
            primary_sources=("Official commodity agencies", "Documented market-data APIs"),
            fallback_sources=("Exchange or index provider pages with accessible terms",),
            reliability_notes="Commodity claims need named contract/index and date.",
            freshness_guidance="Use data-as-of date; avoid mixing spot, futures, and index values.",
        ),
        ContextSourceStrategyItem(
            category=ContextSourceType.SECTOR_CONTEXT,
            primary_sources=("Company filings", "Regulatory data", "Reputable sector research"),
            fallback_sources=("Documented commercial data APIs",),
            reliability_notes="Sector context must stay separate from company fundamentals.",
            freshness_guidance="Prefer dated, accessible sector sources and cite their scope.",
        ),
    )


def _deduplicate_sources(items: tuple[ContextSourceItem, ...]) -> tuple[ContextSourceItem, ...]:
    selected: dict[str, ContextSourceItem] = {}
    for item in items:
        key = _dedupe_key(item)
        current = selected.get(key)
        if current is None or _source_rank(item) > _source_rank(current):
            selected[key] = item
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (item.published_at or item.retrieved_at, item.id),
            reverse=True,
        )
    )


def _dedupe_key(item: ContextSourceItem) -> str:
    if item.source_url:
        return item.source_url.lower()
    return re.sub(r"[^a-z0-9]+", " ", item.title.lower()).strip()


def _source_rank(item: ContextSourceItem) -> tuple[int, datetime, str]:
    reliability_rank = {
        SourceReliability.OFFICIAL: 6,
        SourceReliability.REGULATORY: 6,
        SourceReliability.DOCUMENTED_API: 5,
        SourceReliability.COMPANY_SOURCE: 5,
        SourceReliability.REPUTABLE_NEWS: 4,
        SourceReliability.SECONDARY: 2,
        SourceReliability.UNKNOWN: 1,
    }[item.reliability]
    return (reliability_rank, item.published_at or item.retrieved_at, item.id)


def _scope_items(
    items: tuple[ContextSourceItem, ...],
    scope: ContextScope,
    *,
    company_symbols: tuple[str, ...] = (),
    sector: str | None = None,
    region: str | None = None,
) -> tuple[ContextSourceItem, ...]:
    scoped = tuple(item for item in items if item.scope == scope)
    if company_symbols:
        scoped = tuple(
            item
            for item in scoped
            if not item.company_symbols or set(item.company_symbols).intersection(company_symbols)
        )
    if sector is not None:
        normalized_sector = sector.casefold()
        scoped = tuple(
            item
            for item in scoped
            if item.sector is None or item.sector.casefold() == normalized_sector
        )
    if region is not None:
        scoped = tuple(item for item in scoped if item.region is None or item.region == region)
    return scoped


def _recency(
    item: ContextSourceItem,
    created_at: datetime,
    recent_window: timedelta,
) -> ContextRecency:
    if item.published_at is None:
        return ContextRecency.UNDATED
    if item.published_at > created_at + timedelta(minutes=5):
        return ContextRecency.FUTURE_DATED
    if item.published_at < created_at - recent_window:
        return ContextRecency.STALE
    return ContextRecency.RECENT


def _combined_recency(
    items: tuple[ContextSourceItem, ...],
    created_at: datetime,
    recent_window: timedelta,
) -> ContextRecency:
    recencies = tuple(_recency(item, created_at, recent_window) for item in items)
    if ContextRecency.RECENT in recencies:
        return ContextRecency.RECENT
    if ContextRecency.STALE in recencies:
        return ContextRecency.STALE
    if ContextRecency.FUTURE_DATED in recencies:
        return ContextRecency.FUTURE_DATED
    return ContextRecency.UNDATED


def _scope_warnings(
    items: tuple[ContextSourceItem, ...],
    recency: ContextRecency,
) -> tuple[str, ...]:
    warnings = []
    if recency != ContextRecency.RECENT:
        warnings.append(f"Source recency is {recency.value}.")
    if any(item.reliability == SourceReliability.REPUTABLE_NEWS for item in items):
        warnings.append("News items are contextual sources, not company fundamentals.")
    if any(
        item.reliability in {SourceReliability.SECONDARY, SourceReliability.UNKNOWN}
        for item in items
    ):
        warnings.append("Some context items have lower source reliability.")
    return tuple(dict.fromkeys(warnings))


def _summary(
    scope: ContextScope,
    items: tuple[ContextSourceItem, ...],
    latest: ContextSourceItem,
) -> str:
    date_text = (
        latest.published_at.date().isoformat()
        if latest.published_at is not None
        else latest.retrieved_at.date().isoformat()
    )
    return (
        f"Found {len(items)} reliable {scope.value} source item(s). Latest dated source: "
        f"{date_text}, {latest.source_name}: {latest.title}."
    )


def _confidence(
    items: tuple[ContextSourceItem, ...],
    recency: ContextRecency,
) -> ConfidenceLabel:
    if recency != ContextRecency.RECENT:
        return ConfidenceLabel.LOW
    if any(
        item.reliability in {SourceReliability.OFFICIAL, SourceReliability.REGULATORY}
        for item in items
    ):
        return ConfidenceLabel.HIGH
    return ConfidenceLabel.MEDIUM


def _status(
    findings: tuple[ContextFinding, ...],
    limitations: tuple[str, ...],
) -> ContextAnalysisStatus:
    if all(not finding.source_item_ids for finding in findings):
        return ContextAnalysisStatus.NO_RELIABLE_SOURCES
    if limitations or any(finding.limitations for finding in findings):
        return ContextAnalysisStatus.PARTIAL
    return ContextAnalysisStatus.COMPLETE


def _dedupe_warnings(
    *,
    source_items: tuple[ContextSourceItem, ...],
    deduped_items: tuple[ContextSourceItem, ...],
) -> tuple[str, ...]:
    removed = len(source_items) - len(deduped_items)
    if removed <= 0:
        return ()
    return (f"Deduplicated {removed} repeated or syndicated context source item(s).",)


def _require_text(name: str, value: str) -> str:
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


def _aware_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
