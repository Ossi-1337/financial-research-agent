"""Market data provider contracts, adapters, metrics, and local storage."""

from financial_research_agent.market_data.alpha_vantage import AlphaVantageProvider
from financial_research_agent.market_data.contracts import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataError,
    MarketDataErrorCode,
    MarketDataMetrics,
    MarketDataProvider,
    MarketDataProviderName,
    MarketDataSource,
    MarketQuote,
    MarketSecurity,
)
from financial_research_agent.market_data.metrics import calculate_price_metrics
from financial_research_agent.market_data.store import MarketDataStore
from financial_research_agent.settings import Settings


def create_default_market_data_provider(settings: Settings) -> MarketDataProvider:
    provider = settings.data_sources.market_data_provider
    if provider != MarketDataProviderName.ALPHA_VANTAGE.value:
        raise ValueError(f"Unsupported market data provider: {provider}")
    return AlphaVantageProvider(api_key=settings.data_sources.alpha_vantage_api_key)


__all__ = [
    "AlphaVantageProvider",
    "HistoricalPriceBar",
    "HistoricalPriceResult",
    "MarketDataError",
    "MarketDataErrorCode",
    "MarketDataMetrics",
    "MarketDataProvider",
    "MarketDataProviderName",
    "MarketDataSource",
    "MarketDataStore",
    "MarketQuote",
    "MarketSecurity",
    "calculate_price_metrics",
    "create_default_market_data_provider",
]
