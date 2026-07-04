from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from financial_research_agent.market_data import (
    AlphaVantageProvider,
    HistoricalPriceBar,
    MarketDataError,
    MarketDataErrorCode,
    MarketDataStore,
    MarketSecurity,
    calculate_price_metrics,
)

DAILY_FIXTURE = {
    "Meta Data": {"2. Symbol": "NVO"},
    "Time Series (Daily)": {
        "2026-07-03": {
            "1. open": "101.00",
            "2. high": "106.00",
            "3. low": "100.00",
            "4. close": "105.00",
            "5. volume": "1200",
        },
        "2026-07-02": {
            "1. open": "98.00",
            "2. high": "102.00",
            "3. low": "97.00",
            "4. close": "100.00",
            "5. volume": "1000",
        },
        "2026-07-01": {
            "1. open": "95.00",
            "2. high": "99.00",
            "3. low": "94.00",
            "4. close": "98.00",
            "5. volume": "900",
        },
    },
}

QUOTE_FIXTURE = {
    "Global Quote": {
        "01. symbol": "NVO",
        "05. price": "105.00",
        "06. volume": "1200",
        "07. latest trading day": "2026-07-03",
    }
}


def test_market_security_and_price_bar_contracts_are_immutable() -> None:
    security = MarketSecurity(
        symbol="nvo",
        security_id="sec:ticker:NVO",
        exchange_mic="xnys",
        currency="usd",
    )
    bar = HistoricalPriceBar(
        security=security,
        priced_at=__import__("datetime").date(2026, 7, 3),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("99"),
        close=Decimal("105"),
        volume=1200,
    )

    assert security.symbol == "NVO"
    assert security.exchange_mic == "XNYS"
    assert security.currency == "USD"
    assert bar.id == "NVO:2026-07-03"
    with pytest.raises(FrozenInstanceError):
        security.symbol = "MSFT"  # type: ignore[misc]


def test_calculate_price_metrics_returns_returns_ma_volatility_and_drawdown() -> None:
    security = MarketSecurity(symbol="NVO")
    bars = (
        _bar(security, "2026-07-01", "100"),
        _bar(security, "2026-07-02", "110"),
        _bar(security, "2026-07-03", "99"),
    )

    metrics = calculate_price_metrics(bars, moving_average_windows=(2,))

    assert metrics.latest_close == Decimal("99")
    assert metrics.return_1d == Decimal("-0.1")
    assert metrics.return_total == Decimal("-0.01")
    assert metrics.moving_averages[2] == Decimal("104.5")
    assert metrics.volatility is not None
    assert metrics.max_drawdown == Decimal("-0.1")
    with pytest.raises(TypeError):
        metrics.moving_averages[2] = Decimal("1")  # type: ignore[index]


def test_alpha_vantage_daily_prices_parse_and_include_metadata_warnings() -> None:
    provider = AlphaVantageProvider(
        api_key="test-key",
        http_client=_client_with_json(DAILY_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )
    security = MarketSecurity(symbol="NVO")

    result = asyncio.run(provider.fetch_daily_prices(security))

    assert result.source.provider == "alpha-vantage"
    assert result.source.provider_status == "documented public API"
    assert result.source.data_as_of == __import__("datetime").date(2026, 7, 3)
    assert len(result.bars) == 3
    assert result.bars[-1].close == Decimal("105.00")
    assert result.metrics.latest_close == Decimal("105.00")
    assert "Currency metadata is unavailable" in result.warnings[0]
    assert "Alpha Vantage data may be delayed" in str(result.source.freshness_warning)


def test_alpha_vantage_quote_parses_global_quote() -> None:
    provider = AlphaVantageProvider(
        api_key="test-key",
        http_client=_client_with_json(QUOTE_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    quote = asyncio.run(provider.fetch_quote(MarketSecurity(symbol="NVO")))

    assert quote.price == Decimal("105.00")
    assert quote.volume == 1200
    assert quote.trading_day == __import__("datetime").date(2026, 7, 3)


def test_alpha_vantage_requires_api_key() -> None:
    provider = AlphaVantageProvider(api_key=None, http_client=_client_with_json(DAILY_FIXTURE))

    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.fetch_daily_prices(MarketSecurity(symbol="NVO")))

    assert exc_info.value.code == MarketDataErrorCode.AUTHENTICATION_FAILED


def test_alpha_vantage_maps_rate_limit_message() -> None:
    provider = AlphaVantageProvider(
        api_key="test-key",
        http_client=_client_with_json({"Note": "rate limit"}),
    )

    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.fetch_daily_prices(MarketSecurity(symbol="NVO")))

    assert exc_info.value.code == MarketDataErrorCode.RATE_LIMITED
    assert exc_info.value.retryable is True


def test_alpha_vantage_maps_malformed_daily_payload() -> None:
    provider = AlphaVantageProvider(
        api_key="test-key",
        http_client=_client_with_json({"Meta Data": {}}),
    )

    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.fetch_daily_prices(MarketSecurity(symbol="NVO")))

    assert exc_info.value.code == MarketDataErrorCode.MALFORMED_RESPONSE


def test_market_data_store_persists_and_marks_stale_history(tmp_path) -> None:
    store = MarketDataStore(
        storage_path=tmp_path / "market_data_price_bars.json",
        stale_after=timedelta(days=1),
    )
    provider = AlphaVantageProvider(
        api_key="test-key",
        http_client=_client_with_json(DAILY_FIXTURE),
        now=lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    result = asyncio.run(provider.fetch_daily_prices(MarketSecurity(symbol="NVO")))

    store.save_history(result)
    reloaded = MarketDataStore(
        storage_path=tmp_path / "market_data_price_bars.json",
        stale_after=timedelta(days=1),
    )
    fresh = reloaded.get_history(
        symbol="NVO", provider="alpha-vantage", now=datetime(2026, 7, 1, tzinfo=UTC)
    )
    stale = reloaded.get_history(
        symbol="NVO", provider="alpha-vantage", now=datetime(2026, 7, 4, tzinfo=UTC)
    )

    assert fresh is not None
    assert fresh.bars[-1].close == Decimal("105.00")
    assert stale is not None
    assert "Stored market data is stale" in stale.warnings[-1]
    assert (
        stale.source.freshness_warning
        == "Stored market data is stale; refresh before relying on it."
    )


def _bar(security: MarketSecurity, day: str, close: str) -> HistoricalPriceBar:
    return HistoricalPriceBar(
        security=security,
        priced_at=__import__("datetime").date.fromisoformat(day),
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=1000,
    )


def _client_with_json(payload: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    )
