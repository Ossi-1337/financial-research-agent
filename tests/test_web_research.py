from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from financial_research_agent.persistence import SQLiteDatabase
from financial_research_agent.settings import Settings
from financial_research_agent.web_research import (
    AlphaVantageNewsProvider,
    BoundedWebSourceFetcher,
    BraveSearchProvider,
    InMemoryWebSourceCache,
    SearXNGSearchProvider,
    SQLiteWebSourceCache,
    TavilySearchProvider,
    WebJurisdiction,
    WebResearchError,
    WebResearchRequest,
    WebResearchService,
    WebResearchStatus,
    WebSearchCandidate,
    WebSourceEvidence,
    WebSourceReliability,
    WebSourceType,
    create_web_search_providers,
    ensure_public_https_url,
)
from financial_research_agent.web_research.fetcher import _ensure_public_peer
from financial_research_agent.web_research.policy import SourceClassification

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


class FixtureSearchProvider:
    def __init__(self, candidates: tuple[WebSearchCandidate, ...]) -> None:
        self.candidates = candidates

    async def search(self, request: WebResearchRequest, *, limit: int):
        del request
        return self.candidates[:limit]


class FixtureFetcher:
    async def fetch(self, candidate, *, classification):
        del classification
        return candidate.url, f"Official bounded evidence for {candidate.title}."


class CancelledSearchProvider:
    async def search(self, request: WebResearchRequest, *, limit: int):
        del request, limit
        raise asyncio.CancelledError


def test_brave_provider_parses_results_and_adds_official_domain_filter() -> None:
    seen_query = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_query
        seen_query = request.url.params["q"]
        assert request.headers["x-subscription-token"] == "test-key"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Danish company law",
                            "url": "https://www.retsinformation.dk/example",
                            "description": "Official result",
                        }
                    ]
                }
            },
        )

    provider = BraveSearchProvider(
        "test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(
        provider.search(
            WebResearchRequest(
                query="current A/S accounting rules",
                jurisdiction=WebJurisdiction.DK,
                requires_official_source=True,
            ),
            limit=8,
        )
    )

    assert result[0].provider == "brave"
    assert "site:retsinformation.dk" in seen_query
    assert "test-key" not in seen_query


def test_alpha_vantage_news_preserves_publication_and_ticker_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["function"] == "NEWS_SENTIMENT"
        assert request.url.params["tickers"] == "NVO"
        return httpx.Response(
            200,
            json={
                "feed": [
                    {
                        "title": "Company update",
                        "url": "https://example.com/company-update",
                        "summary": "Documented company-news context.",
                        "source": "Example News",
                        "time_published": "20260801T103000",
                        "ticker_sentiment": [{"ticker": "NVO"}],
                    }
                ]
            },
        )

    provider = AlphaVantageNewsProvider(
        "test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(
        provider.search(WebResearchRequest(query="NVO news", ticker="nvo"), limit=8)
    )

    assert result[0].provider == "alpha-vantage-news"
    assert result[0].published_at == datetime(2026, 8, 1, 10, 30, tzinfo=UTC)
    assert result[0].metadata["ticker_sentiment"] == "NVO"


def test_tavily_provider_uses_bounded_search_without_generated_content() -> None:
    seen_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        seen_body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer tavily-key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Danish company law",
                        "url": "https://www.retsinformation.dk/example",
                        "content": "Official result",
                        "score": 0.9,
                    }
                ]
            },
        )

    provider = TavilySearchProvider(
        "tavily-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(
        provider.search(
            WebResearchRequest(
                query="current A/S accounting rules",
                jurisdiction=WebJurisdiction.DK,
                requires_official_source=True,
            ),
            limit=5,
        )
    )

    assert result[0].provider == "tavily"
    assert seen_body["include_answer"] is False
    assert seen_body["include_raw_content"] is False
    assert seen_body["search_depth"] == "basic"
    assert "retsinformation.dk" in seen_body["include_domains"]


def test_searxng_provider_parses_json_and_filters_non_https_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        assert request.url.params["safesearch"] == "2"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official source",
                        "url": "https://www.sec.gov/example",
                        "content": "SEC result",
                        "engines": ["duckduckgo"],
                    },
                    {
                        "title": "Unsafe transport",
                        "url": "http://example.com/result",
                        "content": "Skipped",
                    },
                ]
            },
        )

    provider = SearXNGSearchProvider(
        "http://searxng:8080",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(provider.search(WebResearchRequest(query="SEC rules"), limit=8))

    assert len(result) == 1
    assert result[0].provider == "searxng"
    assert result[0].metadata["engines"] == "duckduckgo"


def test_searxng_provider_reports_disabled_json_output() -> None:
    provider = SearXNGSearchProvider(
        "http://searxng:8080",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(403))
        ),
    )

    with pytest.raises(WebResearchError, match="JSON output"):
        asyncio.run(provider.search(WebResearchRequest(query="market news"), limit=5))


@pytest.mark.parametrize(
    ("provider", "extra_env", "expected_type"),
    (
        ("brave", {"FRA_BRAVE_SEARCH_API_KEY": "key"}, BraveSearchProvider),
        ("tavily", {"FRA_TAVILY_API_KEY": "key"}, TavilySearchProvider),
        ("searxng", {}, SearXNGSearchProvider),
    ),
)
def test_web_provider_factory_selects_one_discovery_adapter(
    provider: str,
    extra_env: dict[str, str],
    expected_type: type,
) -> None:
    settings = Settings.from_env({"FRA_WEB_SEARCH_PROVIDER": provider, **extra_env})

    providers = create_web_search_providers(settings.data_sources)

    assert len(providers) == 1
    assert isinstance(providers[0], expected_type)


def test_web_research_requires_official_source_for_regulatory_claims() -> None:
    candidate = WebSearchCandidate(
        title="Secondary summary",
        url="https://finance.yahoo.com/example",
        snippet="Secondary",
        provider="brave",
        source_name="Yahoo Finance",
    )
    service = WebResearchService(
        enabled=True,
        search_providers=(FixtureSearchProvider((candidate,)),),
        fetcher=FixtureFetcher(),  # type: ignore[arg-type]
        cache=InMemoryWebSourceCache(),
    )

    result = asyncio.run(
        service.research(
            WebResearchRequest(
                query="Danish A/S rules",
                jurisdiction=WebJurisdiction.DK,
                requires_official_source=True,
            )
        )
    )

    assert result.status == WebResearchStatus.NO_SOURCES
    assert result.sources == ()
    assert result.no_result_reason == (
        "No authoritative source was available for this jurisdiction."
    )


def test_web_research_propagates_cancellation() -> None:
    service = WebResearchService(
        enabled=True,
        search_providers=(CancelledSearchProvider(),),
        fetcher=FixtureFetcher(),  # type: ignore[arg-type]
        cache=InMemoryWebSourceCache(),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.research(WebResearchRequest(query="current market context")))


def test_web_research_accepts_bounded_official_evidence_and_caches_it() -> None:
    candidate = WebSearchCandidate(
        title="Danish company law",
        url="https://www.retsinformation.dk/example?utm_source=test",
        snippet="Official",
        provider="brave",
        source_name="Retsinformation",
    )
    cache = InMemoryWebSourceCache()
    service = WebResearchService(
        enabled=True,
        search_providers=(FixtureSearchProvider((candidate,)),),
        fetcher=FixtureFetcher(),  # type: ignore[arg-type]
        cache=cache,
    )

    result = asyncio.run(
        service.research(
            WebResearchRequest(
                query="Danish A/S rules",
                jurisdiction=WebJurisdiction.DK,
                requires_official_source=True,
            )
        )
    )

    assert result.status == WebResearchStatus.COMPLETE
    assert len(result.sources) == 1
    assert result.sources[0].reliability == WebSourceReliability.REGULATORY
    assert "utm_source" not in result.sources[0].canonical_url
    assert len(result.sources[0].quote) <= 1_500


def test_sqlite_web_cache_round_trip(tmp_path: Path) -> None:
    database = SQLiteDatabase.from_data_dir(tmp_path)
    database.initialize()
    cache = SQLiteWebSourceCache(database)
    source = WebSourceEvidence(
        id="web:test",
        canonical_url="https://www.retsinformation.dk/example",
        title="Official source",
        publisher="Retsinformation",
        quote="Bounded quote",
        source_type=WebSourceType.REGULATORY,
        reliability=WebSourceReliability.REGULATORY,
        provider="brave",
        jurisdiction=WebJurisdiction.DK,
        retrieved_at=NOW,
        expires_at=NOW + timedelta(hours=24),
        content_sha256=hashlib.sha256(b"Bounded quote").hexdigest(),
    )

    cache.save(source)

    assert cache.get(source.canonical_url, now=NOW) == source
    assert cache.get(source.canonical_url, now=NOW + timedelta(days=2)) is None


def test_fetcher_rejects_private_network_url() -> None:
    with pytest.raises(WebResearchError, match=r"not allowed|private"):
        asyncio.run(ensure_public_https_url("https://127.0.0.1/private"))


def test_fetcher_rejects_url_credentials_and_private_connected_peer() -> None:
    from financial_research_agent.web_research.policy import canonicalize_url

    with pytest.raises(ValueError, match="absolute HTTPS"):
        canonicalize_url("https://user:password@example.com/source")

    class PrivateNetworkStream:
        def get_extra_info(self, name: str):
            return ("127.0.0.1", 443) if name == "server_addr" else None

    response = httpx.Response(
        200,
        extensions={"network_stream": PrivateNetworkStream()},
    )
    with pytest.raises(WebResearchError, match="private address"):
        _ensure_public_peer(response)


def test_fetcher_rejects_oversized_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "100"}, content=b"x" * 100)

    fetcher = BoundedWebSourceFetcher(
        max_bytes=20,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        host_validator=lambda _url: _completed(),
    )
    candidate = WebSearchCandidate(
        title="Source",
        url="https://example.com/source",
        snippet="Snippet",
        provider="brave",
        source_name="Example",
    )
    with pytest.raises(WebResearchError, match="size limit"):
        asyncio.run(
            fetcher.fetch(
                candidate,
                classification=SourceClassification(
                    WebSourceType.SECONDARY,
                    WebSourceReliability.SECONDARY,
                    None,
                ),
            )
        )


async def _completed() -> None:
    return None
