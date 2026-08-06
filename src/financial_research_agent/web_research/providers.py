from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from financial_research_agent.web_research.contracts import (
    WebResearchError,
    WebResearchErrorCode,
    WebResearchRequest,
    WebSearchCandidate,
)
from financial_research_agent.web_research.policy import WebSourcePolicy


class BraveSearchProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.search.brave.com/res/v1",
        timeout_seconds: float = 15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = _required(api_key, "api_key")
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(
        self,
        request: WebResearchRequest,
        *,
        limit: int,
    ) -> tuple[WebSearchCandidate, ...]:
        query = request.query
        if request.requires_official_source and request.jurisdiction is not None:
            domains = WebSourcePolicy().official_domains(request.jurisdiction)
            domain_filter = " OR ".join(f"site:{domain}" for domain in domains)
            query = f"{query} ({domain_filter})"
        try:
            response = await self.client.get(
                f"{self.base_url}/web/search",
                params={"q": query, "count": min(limit, 8), "safesearch": "strict"},
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            )
        except httpx.TimeoutException as exc:
            raise WebResearchError(WebResearchErrorCode.TIMEOUT, "Web search timed out.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchError(
                WebResearchErrorCode.PROVIDER_UNAVAILABLE,
                "Web search provider is unavailable.",
            ) from exc
        _raise_for_status(response)
        try:
            payload = response.json()
            items = _mapping(_mapping(payload).get("web")).get("results", ())
            if not isinstance(items, list):
                raise ValueError
            return tuple(
                _brave_candidate(item) for item in items[:limit] if isinstance(item, Mapping)
            )
        except (TypeError, ValueError) as exc:
            raise WebResearchError(
                WebResearchErrorCode.MALFORMED_RESPONSE,
                "Web search provider returned malformed data.",
            ) from exc


class TavilySearchProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.tavily.com",
        timeout_seconds: float = 15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = _required(api_key, "api_key")
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(
        self,
        request: WebResearchRequest,
        *,
        limit: int,
    ) -> tuple[WebSearchCandidate, ...]:
        body: dict[str, object] = {
            "query": request.query,
            "search_depth": "basic",
            "topic": "finance" if request.ticker else "general",
            "max_results": min(limit, 8),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if request.requires_official_source and request.jurisdiction is not None:
            body["include_domains"] = list(WebSourcePolicy().official_domains(request.jurisdiction))
        try:
            response = await self.client.post(
                f"{self.base_url}/search",
                json=body,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        except httpx.TimeoutException as exc:
            raise WebResearchError(WebResearchErrorCode.TIMEOUT, "Web search timed out.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchError(
                WebResearchErrorCode.PROVIDER_UNAVAILABLE,
                "Web search provider is unavailable.",
            ) from exc
        _raise_for_status(response)
        try:
            items = _mapping(response.json()).get("results", ())
            if not isinstance(items, list):
                raise ValueError
            return tuple(
                _tavily_candidate(item) for item in items[:limit] if isinstance(item, Mapping)
            )
        except (TypeError, ValueError) as exc:
            raise WebResearchError(
                WebResearchErrorCode.MALFORMED_RESPONSE,
                "Web search provider returned malformed data.",
            ) from exc


class SearXNGSearchProvider:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = _required(base_url, "base_url").rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(
        self,
        request: WebResearchRequest,
        *,
        limit: int,
    ) -> tuple[WebSearchCandidate, ...]:
        query = request.query
        if request.requires_official_source and request.jurisdiction is not None:
            domains = WebSourcePolicy().official_domains(request.jurisdiction)
            query = f"{query} ({' OR '.join(f'site:{domain}' for domain in domains)})"
        try:
            response = await self.client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "safesearch": "2"},
                headers={"Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise WebResearchError(WebResearchErrorCode.TIMEOUT, "Web search timed out.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchError(
                WebResearchErrorCode.PROVIDER_UNAVAILABLE,
                "Web search provider is unavailable.",
            ) from exc
        if response.status_code == 403:
            raise WebResearchError(
                WebResearchErrorCode.MALFORMED_RESPONSE,
                "SearXNG JSON output is not enabled on the configured instance.",
            )
        _raise_for_status(response)
        try:
            items = _mapping(response.json()).get("results", ())
            if not isinstance(items, list):
                raise ValueError
            candidates = (_searxng_candidate(item) for item in items if isinstance(item, Mapping))
            return tuple(candidate for candidate in candidates if candidate is not None)[:limit]
        except (TypeError, ValueError) as exc:
            raise WebResearchError(
                WebResearchErrorCode.MALFORMED_RESPONSE,
                "Web search provider returned malformed data.",
            ) from exc


class AlphaVantageNewsProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://www.alphavantage.co/query",
        timeout_seconds: float = 15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = _required(api_key, "api_key")
        self.base_url = base_url
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(
        self,
        request: WebResearchRequest,
        *,
        limit: int,
    ) -> tuple[WebSearchCandidate, ...]:
        if request.ticker is None:
            return ()
        try:
            response = await self.client.get(
                self.base_url,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": request.ticker,
                    "sort": "RELEVANCE",
                    "limit": min(limit, 8),
                    "apikey": self.api_key,
                },
            )
        except httpx.TimeoutException as exc:
            raise WebResearchError(WebResearchErrorCode.TIMEOUT, "News search timed out.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchError(
                WebResearchErrorCode.PROVIDER_UNAVAILABLE,
                "News provider is unavailable.",
            ) from exc
        _raise_for_status(response)
        try:
            payload = _mapping(response.json())
            if payload.get("Information") or payload.get("Note"):
                raise WebResearchError(
                    WebResearchErrorCode.RATE_LIMITED,
                    "News provider limit reached.",
                )
            feed = payload.get("feed", ())
            if not isinstance(feed, list):
                raise ValueError
            return tuple(
                _alpha_candidate(item) for item in feed[:limit] if isinstance(item, Mapping)
            )
        except WebResearchError:
            raise
        except (TypeError, ValueError) as exc:
            raise WebResearchError(
                WebResearchErrorCode.MALFORMED_RESPONSE,
                "News provider returned malformed data.",
            ) from exc


def _brave_candidate(value: Mapping[object, object]) -> WebSearchCandidate:
    url = str(value.get("url", ""))
    return WebSearchCandidate(
        title=str(value.get("title", "")),
        url=url,
        snippet=str(value.get("description", "")),
        provider="brave",
        source_name=urlsplit(url).hostname or "web",
    )


def _tavily_candidate(value: Mapping[object, object]) -> WebSearchCandidate:
    url = str(value.get("url", ""))
    content = str(value.get("content", "")).strip()
    return WebSearchCandidate(
        title=str(value.get("title", "")),
        url=url,
        snippet=content or str(value.get("title", "")),
        provider="tavily",
        source_name=urlsplit(url).hostname or "web",
        published_at=_parse_optional_datetime(value.get("published_date")),
        metadata={"score": str(value.get("score", "unavailable"))},
    )


def _searxng_candidate(
    value: Mapping[object, object],
) -> WebSearchCandidate | None:
    url = str(value.get("url", ""))
    if not url.lower().startswith("https://"):
        return None
    title = str(value.get("title", "")).strip()
    snippet = str(value.get("content", "")).strip()
    if not title or not snippet:
        return None
    engines = value.get("engines", ())
    engine_names = (
        ",".join(str(item) for item in engines)
        if isinstance(engines, list)
        else str(value.get("engine", "unknown"))
    )
    return WebSearchCandidate(
        title=title,
        url=url,
        snippet=snippet,
        provider="searxng",
        source_name=urlsplit(url).hostname or "web",
        published_at=_parse_optional_datetime(
            value.get("publishedDate") or value.get("published_date")
        ),
        metadata={"engines": engine_names or "unknown"},
    )


def _alpha_candidate(value: Mapping[object, object]) -> WebSearchCandidate:
    published = str(value.get("time_published", "")).strip()
    return WebSearchCandidate(
        title=str(value.get("title", "")),
        url=str(value.get("url", "")),
        snippet=str(value.get("summary", "")),
        provider="alpha-vantage-news",
        source_name=str(value.get("source", "Alpha Vantage News")),
        published_at=(
            datetime.strptime(published, "%Y%m%dT%H%M%S").replace(tzinfo=UTC) if published else None
        ),
        metadata={"ticker_sentiment": _ticker_sentiment(value)},
    )


def _ticker_sentiment(value: Mapping[object, object]) -> str:
    items = value.get("ticker_sentiment", ())
    if not isinstance(items, list):
        return "unavailable"
    return (
        ",".join(
            str(item.get("ticker", ""))
            for item in items
            if isinstance(item, Mapping) and item.get("ticker")
        )
        or "unavailable"
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise WebResearchError(
            WebResearchErrorCode.AUTHENTICATION_FAILED,
            "Provider authentication failed.",
        )
    if response.status_code in {429, 432, 433}:
        raise WebResearchError(WebResearchErrorCode.RATE_LIMITED, "Provider rate limit reached.")
    if response.status_code >= 500:
        raise WebResearchError(
            WebResearchErrorCode.PROVIDER_UNAVAILABLE,
            "Provider is unavailable.",
        )
    if response.status_code >= 400:
        raise WebResearchError(
            WebResearchErrorCode.MALFORMED_RESPONSE,
            "Provider rejected the request.",
        )


def _mapping(value: object) -> Mapping[object, object]:
    return value if isinstance(value, Mapping) else {}


def _required(value: str, name: str) -> str:
    if not (text := value.strip()):
        raise ValueError(f"{name} is required")
    return text


def _parse_optional_datetime(value: object) -> datetime | None:
    if value is None or not (text := str(value).strip()):
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
