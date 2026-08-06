from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from time import perf_counter_ns
from urllib.parse import urlsplit

from financial_research_agent.context_analysis import (
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    SourceReliability,
)
from financial_research_agent.web_research.contracts import (
    WebResearchError,
    WebResearchRequest,
    WebResearchResult,
    WebResearchStatus,
    WebSearchCandidate,
    WebSearchProvider,
    WebSourceCache,
    WebSourceEvidence,
    WebSourceReliability,
    WebSourceType,
)
from financial_research_agent.web_research.fetcher import BoundedWebSourceFetcher
from financial_research_agent.web_research.policy import WebSourcePolicy, canonicalize_url


class WebResearchService:
    def __init__(
        self,
        *,
        enabled: bool,
        search_providers: Iterable[WebSearchProvider],
        fetcher: BoundedWebSourceFetcher,
        cache: WebSourceCache,
        max_results: int = 8,
        max_sources: int = 5,
        news_ttl: timedelta = timedelta(minutes=60),
        regulatory_ttl: timedelta = timedelta(hours=24),
        policy: WebSourcePolicy | None = None,
    ) -> None:
        if max_results <= 0 or max_results > 8:
            raise ValueError("max_results must be between 1 and 8")
        if max_sources <= 0 or max_sources > 5:
            raise ValueError("max_sources must be between 1 and 5")
        self.enabled = enabled
        self.search_providers = tuple(search_providers)
        self.fetcher = fetcher
        self.cache = cache
        self.max_results = max_results
        self.max_sources = max_sources
        self.news_ttl = news_ttl
        self.regulatory_ttl = regulatory_ttl
        self.policy = policy or WebSourcePolicy()

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.search_providers)

    async def research(self, request: WebResearchRequest) -> WebResearchResult:
        started_ns = perf_counter_ns()
        if not self.enabled:
            return _empty_result(
                WebResearchStatus.UNAVAILABLE,
                "Web research is disabled.",
                started_ns,
            )
        if not self.search_providers:
            return _empty_result(
                WebResearchStatus.UNAVAILABLE,
                "No web search provider is configured.",
                started_ns,
            )
        outcomes = await asyncio.gather(
            *(
                provider.search(request, limit=self.max_results)
                for provider in self.search_providers
            ),
            return_exceptions=True,
        )
        candidates: list[WebSearchCandidate] = []
        warnings: list[str] = []
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, WebResearchError):
                warnings.append(f"Web provider warning: {outcome.code.value}.")
            elif isinstance(outcome, Exception):
                warnings.append("Web provider warning: provider_unavailable.")
            else:
                candidates.extend(outcome)
        ranked = self._rank_candidates(candidates, request)
        sources: list[WebSourceEvidence] = []
        total_chars = 0
        now = datetime.now(UTC)
        for candidate in ranked[: self.max_results]:
            try:
                canonical_url = canonicalize_url(candidate.url)
                cached = self.cache.get(canonical_url, now=now)
                if cached is not None:
                    source = cached
                else:
                    classification = self.policy.classify(
                        candidate,
                        requested_jurisdiction=request.jurisdiction,
                    )
                    fetched_url, text = await self.fetcher.fetch(
                        candidate,
                        classification=classification,
                    )
                    source = self._build_source(
                        candidate,
                        canonical_url=canonicalize_url(fetched_url),
                        text=text,
                        request=request,
                        retrieved_at=now,
                    )
                    self.cache.save(source)
            except (ValueError, WebResearchError) as exc:
                code = exc.code.value if isinstance(exc, WebResearchError) else "unsafe_url"
                warnings.append(f"Skipped web candidate: {code}.")
                continue
            available = 5_000 - total_chars
            if available <= 0:
                break
            if len(source.quote) > available:
                source = _with_quote(source, source.quote[:available].rstrip())
            sources.append(source)
            total_chars += len(source.quote)
            if len(sources) >= self.max_sources:
                break
        if request.requires_official_source and not any(
            source.reliability in {WebSourceReliability.OFFICIAL, WebSourceReliability.REGULATORY}
            and (request.jurisdiction is None or source.jurisdiction == request.jurisdiction)
            for source in sources
        ):
            return WebResearchResult(
                status=WebResearchStatus.NO_SOURCES,
                sources=(),
                warnings=tuple(dict.fromkeys(warnings)),
                no_result_reason="No authoritative source was available for this jurisdiction.",
                duration_ms=_elapsed_ms(started_ns),
            )
        status = (
            WebResearchStatus.COMPLETE
            if sources and not warnings
            else WebResearchStatus.PARTIAL
            if sources
            else WebResearchStatus.NO_SOURCES
        )
        return WebResearchResult(
            status=status,
            sources=tuple(sources),
            warnings=tuple(dict.fromkeys(warnings)),
            no_result_reason=None if sources else "No acceptable web evidence was found.",
            duration_ms=_elapsed_ms(started_ns),
        )

    def _rank_candidates(
        self,
        candidates: Iterable[WebSearchCandidate],
        request: WebResearchRequest,
    ) -> tuple[WebSearchCandidate, ...]:
        unique: dict[str, WebSearchCandidate] = {}
        for candidate in candidates:
            try:
                unique.setdefault(canonicalize_url(candidate.url), candidate)
            except ValueError:
                continue
        return tuple(
            sorted(
                unique.values(),
                key=lambda candidate: _candidate_rank(
                    self.policy.classify(
                        candidate,
                        requested_jurisdiction=request.jurisdiction,
                    )
                ),
                reverse=True,
            )
        )

    def _build_source(
        self,
        candidate: WebSearchCandidate,
        *,
        canonical_url: str,
        text: str,
        request: WebResearchRequest,
        retrieved_at: datetime,
    ) -> WebSourceEvidence:
        classification = self.policy.classify(
            candidate,
            requested_jurisdiction=request.jurisdiction,
        )
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ttl = (
            self.news_ttl
            if classification.source_type == WebSourceType.NEWS
            else self.regulatory_ttl
        )
        return WebSourceEvidence(
            id=f"web:{hashlib.sha256(canonical_url.encode('utf-8')).hexdigest()[:24]}",
            canonical_url=canonical_url,
            title=candidate.title,
            publisher=candidate.source_name or (urlsplit(canonical_url).hostname or "web"),
            quote=text[:1_500].rstrip(),
            source_type=classification.source_type,
            reliability=classification.reliability,
            provider=candidate.provider,
            jurisdiction=classification.jurisdiction,
            published_at=candidate.published_at,
            retrieved_at=retrieved_at,
            expires_at=retrieved_at + ttl,
            content_sha256=content_hash,
            warnings=classification.warnings,
            metadata=candidate.metadata,
        )


def to_context_source_items(
    result: WebResearchResult,
    *,
    company_symbols: tuple[str, ...] = (),
) -> tuple[ContextSourceItem, ...]:
    return tuple(
        ContextSourceItem(
            id=source.id,
            title=source.title,
            summary=source.quote,
            source_url=source.canonical_url,
            source_name=source.publisher,
            source_type=_context_source_type(source.source_type),
            reliability=SourceReliability(source.reliability.value),
            scope=ContextScope.COMPANY if company_symbols else ContextScope.MACRO,
            retrieved_at=source.retrieved_at,
            published_at=source.published_at,
            company_symbols=company_symbols,
            region=source.jurisdiction.value if source.jurisdiction else None,
            topics=(source.source_type.value,),
            metadata={
                **dict(source.metadata),
                "content_sha256": source.content_sha256,
                "provider": source.provider,
                "expires_at": source.expires_at.isoformat(),
            },
        )
        for source in result.sources
    )


def _context_source_type(source_type: WebSourceType) -> ContextSourceType:
    if source_type == WebSourceType.NEWS:
        return ContextSourceType.COMPANY_NEWS
    if source_type == WebSourceType.MACRO:
        return ContextSourceType.MACRO_INDICATOR
    if source_type == WebSourceType.COMPANY:
        return ContextSourceType.COMPANY_EVENT
    return ContextSourceType.SECTOR_CONTEXT


def _candidate_rank(classification) -> int:
    return {
        WebSourceReliability.REGULATORY: 7,
        WebSourceReliability.OFFICIAL: 7,
        WebSourceReliability.COMPANY_SOURCE: 6,
        WebSourceReliability.DOCUMENTED_API: 5,
        WebSourceReliability.REPUTABLE_NEWS: 4,
        WebSourceReliability.SECONDARY: 2,
        WebSourceReliability.UNKNOWN: 1,
    }[classification.reliability]


def _with_quote(source: WebSourceEvidence, quote: str) -> WebSourceEvidence:
    payload = source.to_dict()
    payload["quote"] = quote
    return WebSourceEvidence.from_dict(payload)


def _empty_result(
    status: WebResearchStatus,
    reason: str,
    started_ns: int,
) -> WebResearchResult:
    return WebResearchResult(
        status=status,
        sources=(),
        no_result_reason=reason,
        duration_ms=_elapsed_ms(started_ns),
    )


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)
