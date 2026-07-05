from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.llm.registry import create_offline_provider_registry
from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataSource,
    MarketDataStore,
    MarketSecurity,
    calculate_price_metrics,
)
from financial_research_agent.settings import Settings
from financial_research_agent.stock_analysis import (
    ConfidenceLabel,
    StockChartPoint,
    StockPriceAnalysisAgent,
    StockPriceAnalysisSecurity,
    StockPriceAnalysisStatus,
    StockPriceFinding,
    StockTrendDirection,
)
from financial_research_agent.web import ChatSessionStore, create_app

NOW = datetime(2026, 7, 5, tzinfo=UTC)


def test_stock_price_contracts_are_immutable_and_validate_shape() -> None:
    security = StockPriceAnalysisSecurity(symbol=" nvo ", currency="dkk")
    point = StockChartPoint(
        priced_at=date(2026, 7, 3),
        close=Decimal("105"),
        adjusted_close=Decimal("104"),
        volume=1200,
        moving_averages={5: Decimal("101")},
    )

    assert security.symbol == "NVO"
    assert security.currency == "DKK"
    assert point.to_dict()["moving_averages"]["5"] == "101"
    with pytest.raises(FrozenInstanceError):
        security.symbol = "TSLA"  # type: ignore[misc]
    with pytest.raises(ValueError, match="metric_names or limitations"):
        StockPriceFinding(
            id="finding:invalid",
            section="trend",
            title="Invalid",
            summary="Invalid",
            confidence=ConfidenceLabel.HIGH,
        )


def test_stock_price_analysis_agent_outputs_metrics_findings_and_chart_data() -> None:
    store = MarketDataStore()
    store.save_history(_history("NVO", (100, 101, 103, 102, 104, 108), adjusted=True))
    store.save_history(_history("SPY", (400, 402, 401, 405, 407, 410)))
    agent = StockPriceAnalysisAgent(
        market_data_store=store,
        market_data_provider="alpha-vantage",
        now=lambda: NOW,
    )

    result = agent.analyze(
        StockPriceAnalysisSecurity(symbol="NVO", security_id="fixture:security:nvo"),
        benchmark_symbol="SPY",
    )

    assert result.status == StockPriceAnalysisStatus.COMPLETE
    assert result.security.symbol == "NVO"
    assert result.benchmark_security is not None
    assert result.benchmark_security.symbol == "SPY"
    assert len(result.chart_series) == 2
    assert result.chart_series[0].points[-1].moving_averages[5] == Decimal("103.6")
    assert "Adjusted close values are present" in " ".join(result.warnings)
    assert "delayed" in " ".join(result.warnings)
    assert _metric(result, "latest_close").value == Decimal("108")
    assert _metric(result, "relative_return_vs_benchmark").unit == "ratio"
    assert {finding.section.value for finding in result.findings} == {
        "recent_performance",
        "trend",
        "volatility",
        "drawdown",
        "volume",
        "benchmark_comparison",
    }
    assert _finding(result.findings, "trend").trend in {
        StockTrendDirection.UP,
        StockTrendDirection.MIXED,
    }


def test_stock_price_analysis_agent_marks_stale_data_and_missing_benchmark_partial() -> None:
    store = MarketDataStore(stale_after=timedelta(days=1))
    store.save_history(
        _history(
            "NVO",
            (100, 99, 98, 97, 96, 95),
            retrieved_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
    )
    agent = StockPriceAnalysisAgent(
        market_data_store=store,
        market_data_provider="alpha-vantage",
        now=lambda: NOW,
    )

    result = agent.analyze(StockPriceAnalysisSecurity(symbol="NVO"), benchmark_symbol="SPY")

    assert result.status == StockPriceAnalysisStatus.PARTIAL
    assert "Stored market data is stale" in " ".join(result.warnings)
    assert "No stored benchmark market data" in " ".join(result.limitations)
    assert result.benchmark_security is None


def test_stock_price_analysis_agent_reports_no_data_without_metrics() -> None:
    agent = StockPriceAnalysisAgent(market_data_store=MarketDataStore(), now=lambda: NOW)

    result = agent.analyze(StockPriceAnalysisSecurity(symbol="NVO"))

    assert result.status == StockPriceAnalysisStatus.NO_DATA
    assert result.metrics == ()
    assert result.chart_series == ()
    assert result.limitations
    assert result.findings[0].confidence == ConfidenceLabel.UNKNOWN


def test_stock_price_analysis_agent_degrades_on_store_failure() -> None:
    agent = StockPriceAnalysisAgent(
        market_data_store=FailingMarketStore("market data file is unreadable"),
        now=lambda: NOW,
    )

    result = agent.analyze(StockPriceAnalysisSecurity(symbol="NVO"))

    assert result.status == StockPriceAnalysisStatus.NO_DATA
    assert "market data file is unreadable" in " ".join(result.limitations)


def test_stock_price_analysis_endpoint_uses_stored_market_data() -> None:
    store = MarketDataStore()
    store.save_history(_history("NVO", (100, 101, 103, 102, 104, 108)))
    client = TestClient(
        create_app(
            settings=Settings.from_env({}),
            registry=create_offline_provider_registry(),
            session_store=ChatSessionStore(),
            market_data_store=store,
        )
    )

    response = client.post(
        "/api/stock-price-analysis",
        json={
            "symbol": "NVO",
            "security_id": "fixture:security:nvo",
            "exchange_mic": "XNYS",
            "currency": "USD",
        },
    )
    payload = response.json()["analysis"]

    assert response.status_code == 200
    assert payload["security"]["symbol"] == "NVO"
    assert payload["status"] == "complete"
    assert payload["metrics"][0]["name"] == "observation_count"
    assert payload["chart_series"][0]["points"][-1]["close"] == "108"
    assert payload["primary_source"]["provider"] == "alpha-vantage"
    assert payload["no_trading_signal_notice"].startswith("This stock price analysis")


def _history(
    symbol: str,
    closes: tuple[int, ...],
    *,
    adjusted: bool = False,
    retrieved_at: datetime = NOW,
) -> HistoricalPriceResult:
    security = MarketSecurity(
        symbol=symbol,
        security_id=f"fixture:security:{symbol.lower()}",
        currency="USD",
    )
    start_date = date(2026, 6, 28)
    bars = tuple(
        HistoricalPriceBar(
            security=security,
            priced_at=start_date + timedelta(days=index),
            open=Decimal(close - 1),
            high=Decimal(close + 1),
            low=Decimal(close - 2),
            close=Decimal(close),
            adjusted_close=(Decimal(close) - Decimal("0.5")) if adjusted else None,
            volume=1000 + (index * 100),
        )
        for index, close in enumerate(closes)
    )
    source = MarketDataSource(
        provider="alpha-vantage",
        provider_status="test fixture",
        source_url=f"https://example.invalid/market-data/{symbol}",
        retrieved_at=retrieved_at,
        data_as_of=bars[-1].priced_at,
        attribution="test fixture",
    )
    return HistoricalPriceResult(
        security=security,
        bars=bars,
        source=source,
        metrics=calculate_price_metrics(bars),
        warnings=("TEST TOOL OUTPUT market data fixture.",),
    )


def _metric(result, name: str):
    return next(metric for metric in result.metrics if metric.name == name)


def _finding(findings: tuple[StockPriceFinding, ...], section: str) -> StockPriceFinding:
    return next(finding for finding in findings if finding.section.value == section)


class FailingMarketStore:
    def __init__(self, message: str) -> None:
        self._message = message

    def get_history(self, **_kwargs):
        raise ValueError(self._message)
