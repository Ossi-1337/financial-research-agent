"""Company search and entity resolution contracts and providers."""

from datetime import timedelta

from financial_research_agent.entities.contracts import (
    CompanySearchCandidate,
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchProvider,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
    SourceMetadata,
)
from financial_research_agent.entities.sec import SECCompanyTickerProvider, SECCompanyTickerRecord
from financial_research_agent.settings import Settings


def create_default_company_search_provider(settings: Settings) -> CompanySearchProvider:
    provider = settings.data_sources.company_lookup_provider
    if provider != "sec":
        raise ValueError(f"Unsupported company lookup provider: {provider}")
    return SECCompanyTickerProvider(
        cache_path=settings.local_paths.cache_dir / "sec_company_tickers.json",
        cache_ttl=timedelta(days=settings.data_sources.company_lookup_cache_ttl_days),
        user_agent=settings.data_sources.sec_user_agent,
    )


__all__ = [
    "CompanySearchCandidate",
    "CompanySearchError",
    "CompanySearchErrorCode",
    "CompanySearchProvider",
    "CompanySearchResult",
    "CompanySearchStatus",
    "EntityIdentifier",
    "EntityIdentifierType",
    "ResolvedCompany",
    "ResolvedSecurity",
    "SECCompanyTickerProvider",
    "SECCompanyTickerRecord",
    "SourceMetadata",
    "create_default_company_search_provider",
]
