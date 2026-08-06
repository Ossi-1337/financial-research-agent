from __future__ import annotations

from financial_research_agent.settings import DataSourceSettings
from financial_research_agent.web_research.contracts import WebSearchProvider
from financial_research_agent.web_research.providers import (
    AlphaVantageNewsProvider,
    BraveSearchProvider,
    SearXNGSearchProvider,
    TavilySearchProvider,
)


def create_web_search_providers(
    settings: DataSourceSettings,
) -> tuple[WebSearchProvider, ...]:
    providers: list[WebSearchProvider] = []
    if settings.web_search_provider == "brave" and settings.brave_search_api_key:
        providers.append(
            BraveSearchProvider(
                settings.brave_search_api_key,
                timeout_seconds=settings.web_search_timeout_seconds,
            )
        )
    elif settings.web_search_provider == "tavily" and settings.tavily_api_key:
        providers.append(
            TavilySearchProvider(
                settings.tavily_api_key,
                base_url=settings.tavily_base_url,
                timeout_seconds=settings.web_search_timeout_seconds,
            )
        )
    elif settings.web_search_provider == "searxng":
        providers.append(
            SearXNGSearchProvider(
                settings.searxng_base_url,
                timeout_seconds=settings.web_search_timeout_seconds,
            )
        )
    if settings.alpha_vantage_api_key:
        providers.append(
            AlphaVantageNewsProvider(
                settings.alpha_vantage_api_key,
                timeout_seconds=settings.web_search_timeout_seconds,
            )
        )
    return tuple(providers)
