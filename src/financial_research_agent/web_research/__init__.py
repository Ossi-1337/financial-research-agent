"""Bounded web, news, and regulatory research contracts."""

from financial_research_agent.web_research.contracts import (
    CitedContextAnswer,
    WebJurisdiction,
    WebResearchError,
    WebResearchErrorCode,
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
from financial_research_agent.web_research.factory import create_web_search_providers
from financial_research_agent.web_research.fetcher import (
    BoundedWebSourceFetcher,
    ensure_public_https_url,
)
from financial_research_agent.web_research.policy import WebSourcePolicy, canonicalize_url
from financial_research_agent.web_research.providers import (
    AlphaVantageNewsProvider,
    BraveSearchProvider,
    SearXNGSearchProvider,
    TavilySearchProvider,
)
from financial_research_agent.web_research.service import (
    WebResearchService,
    to_context_source_items,
)
from financial_research_agent.web_research.store import (
    InMemoryWebSourceCache,
    SQLiteWebSourceCache,
)

__all__ = [
    "AlphaVantageNewsProvider",
    "BoundedWebSourceFetcher",
    "BraveSearchProvider",
    "CitedContextAnswer",
    "InMemoryWebSourceCache",
    "SQLiteWebSourceCache",
    "SearXNGSearchProvider",
    "TavilySearchProvider",
    "WebJurisdiction",
    "WebResearchError",
    "WebResearchErrorCode",
    "WebResearchRequest",
    "WebResearchResult",
    "WebResearchService",
    "WebResearchStatus",
    "WebSearchCandidate",
    "WebSearchProvider",
    "WebSourceCache",
    "WebSourceEvidence",
    "WebSourcePolicy",
    "WebSourceReliability",
    "WebSourceType",
    "canonicalize_url",
    "create_web_search_providers",
    "ensure_public_https_url",
    "to_context_source_items",
]
