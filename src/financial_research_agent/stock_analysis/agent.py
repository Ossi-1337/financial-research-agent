from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol
from uuid import uuid4

from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataMetrics,
)
from financial_research_agent.stock_analysis.contracts import (
    NO_TRADING_SIGNAL_NOTICE,
    ConfidenceLabel,
    StockChartPoint,
    StockChartSeries,
    StockPriceAnalysisResult,
    StockPriceAnalysisSection,
    StockPriceAnalysisSecurity,
    StockPriceAnalysisStatus,
    StockPriceFinding,
    StockPriceMetric,
    StockTrendDirection,
)

MARKET_DATA_DELAY_WARNING = (
    "Stored market data may be delayed, end-of-day, provider-limited, or outside normal "
    "market hours; do not treat it as a live trading feed."
)
SPLIT_ADJUSTMENT_WARNING = (
    "Adjusted close values are present. Verify split/dividend adjustment policy before "
    "comparing long time ranges."
)


class MarketDataResultStore(Protocol):
    def get_history(
        self,
        *,
        symbol: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> HistoricalPriceResult | None: ...


class StockPriceAnalysisAgent:
    """Deterministic stock-price analysis over already stored market data."""

    def __init__(
        self,
        *,
        market_data_store: MarketDataResultStore,
        market_data_provider: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._market_data_store = market_data_store
        self._market_data_provider = market_data_provider
        self._now = now or (lambda: datetime.now(UTC))

    def analyze(
        self,
        security: StockPriceAnalysisSecurity,
        *,
        benchmark_symbol: str | None = None,
    ) -> StockPriceAnalysisResult:
        created_at = _aware_now(self._now())
        warnings = [MARKET_DATA_DELAY_WARNING]
        limitations: list[str] = []
        history = self._stored_history(security.symbol, created_at, limitations)
        benchmark_history = None
        benchmark_security = None

        if benchmark_symbol is not None:
            benchmark = StockPriceAnalysisSecurity(symbol=benchmark_symbol)
            benchmark_history = self._stored_history(benchmark.symbol, created_at, limitations)
            if benchmark_history is None:
                limitations.append(
                    f"No stored benchmark market data was available for {benchmark.symbol}."
                )
            else:
                benchmark_security = _security_from_history(benchmark_history)
                warnings.extend(_source_warnings(benchmark_history))

        if history is None:
            limitations.append(
                f"No stored market data was available for {security.symbol}. Fetch daily "
                "historical prices before running stock price analysis."
            )
            return StockPriceAnalysisResult(
                id=f"stock_price_analysis_{security.symbol}_{uuid4().hex}",
                security=security,
                status=StockPriceAnalysisStatus.NO_DATA,
                created_at=created_at,
                metrics=(),
                findings=(
                    _limited_finding(
                        StockPriceAnalysisSection.RECENT_PERFORMANCE,
                        "Recent Performance",
                        "Stored price history is unavailable.",
                    ),
                ),
                chart_series=(),
                warnings=tuple(dict.fromkeys(warnings)),
                limitations=tuple(dict.fromkeys(limitations)),
                no_trading_signal_notice=NO_TRADING_SIGNAL_NOTICE,
            )

        selected_security = _security_from_history(history, fallback=security)
        warnings.extend(_source_warnings(history))
        bars = _sorted_bars(history.bars)
        benchmark_bars = _sorted_bars(benchmark_history.bars) if benchmark_history else ()
        metrics = _metrics(history, benchmark_history)
        findings = (
            _recent_performance_finding(history.metrics),
            _trend_finding(history.metrics),
            _volatility_finding(history.metrics),
            _drawdown_finding(history.metrics),
            _volume_finding(bars),
            *(
                (_benchmark_finding(history.metrics, benchmark_history.metrics),)
                if benchmark_history is not None
                else ()
            ),
        )
        status = _analysis_status(findings=findings, limitations=limitations)
        return StockPriceAnalysisResult(
            id=f"stock_price_analysis_{selected_security.symbol}_{uuid4().hex}",
            security=selected_security,
            status=status,
            created_at=created_at,
            metrics=metrics,
            findings=findings,
            chart_series=tuple(
                series
                for series in (
                    _chart_series("primary", selected_security.symbol, bars),
                    (
                        _chart_series("benchmark", benchmark_security.symbol, benchmark_bars)
                        if benchmark_security is not None
                        else None
                    ),
                )
                if series is not None
            ),
            primary_source=history.source,
            benchmark_security=benchmark_security,
            benchmark_source=benchmark_history.source if benchmark_history is not None else None,
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=tuple(dict.fromkeys(limitations)),
            no_trading_signal_notice=NO_TRADING_SIGNAL_NOTICE,
        )

    def _stored_history(
        self,
        symbol: str,
        now: datetime,
        limitations: list[str],
    ) -> HistoricalPriceResult | None:
        try:
            return self._market_data_store.get_history(
                symbol=symbol,
                provider=self._market_data_provider,
                now=now,
            )
        except Exception as exc:
            limitations.append(f"Market data store failed for {symbol}: {exc}")
            return None


def _metrics(
    history: HistoricalPriceResult,
    benchmark_history: HistoricalPriceResult | None,
) -> tuple[StockPriceMetric, ...]:
    bars = _sorted_bars(history.bars)
    values: list[StockPriceMetric] = [
        _metric(
            "observation_count",
            Decimal(len(bars)),
            "count",
            "Number of stored daily price observations used by this analysis.",
        )
    ]
    latest = bars[-1] if bars else None
    if latest is not None:
        values.extend(
            (
                _metric("latest_close", latest.close, "price", "Latest stored daily close."),
                _metric("latest_volume", Decimal(latest.volume), "shares", "Latest stored volume."),
            )
        )
    _append_market_metrics(values, history.metrics)
    return_5d = _window_return(bars, 5)
    if return_5d is not None:
        values.append(_metric("return_5d", return_5d, "ratio", "Five-trading-day return."))
    return_20d = _window_return(bars, 20)
    if return_20d is not None:
        values.append(_metric("return_20d", return_20d, "ratio", "Twenty-trading-day return."))
    volume_average = _average_volume(bars)
    if volume_average is not None:
        values.append(
            _metric("average_volume", volume_average, "shares", "Average volume in stored bars.")
        )
        if latest is not None and volume_average != 0:
            values.append(
                _metric(
                    "latest_volume_vs_average",
                    (Decimal(latest.volume) - volume_average) / volume_average,
                    "ratio",
                    "Latest volume relative to average volume in stored bars.",
                )
            )
    if benchmark_history is not None and benchmark_history.metrics.return_total is not None:
        benchmark_return = benchmark_history.metrics.return_total
        values.append(
            _metric(
                "benchmark_return_total",
                benchmark_return,
                "ratio",
                "Benchmark total return over its stored range.",
            )
        )
        if history.metrics.return_total is not None:
            values.append(
                _metric(
                    "relative_return_vs_benchmark",
                    history.metrics.return_total - benchmark_return,
                    "ratio",
                    "Security total return minus benchmark total return.",
                )
            )
    return tuple(values)


def _append_market_metrics(values: list[StockPriceMetric], metrics: MarketDataMetrics) -> None:
    if metrics.return_1d is not None:
        values.append(_metric("return_1d", metrics.return_1d, "ratio", "One-day close return."))
    if metrics.return_total is not None:
        values.append(
            _metric("return_total", metrics.return_total, "ratio", "Total return over stored bars.")
        )
    for window, value in metrics.moving_averages.items():
        values.append(
            _metric(
                f"moving_average_{window}",
                value,
                "price",
                f"{window}-bar moving average of daily close.",
            )
        )
    if metrics.volatility is not None:
        values.append(
            _metric(
                "annualized_volatility",
                metrics.volatility,
                "ratio",
                "Annualized volatility calculated from daily close returns.",
            )
        )
    if metrics.max_drawdown is not None:
        values.append(
            _metric(
                "max_drawdown",
                metrics.max_drawdown,
                "ratio",
                "Maximum peak-to-trough drawdown over stored closes.",
            )
        )


def _recent_performance_finding(metrics: MarketDataMetrics) -> StockPriceFinding:
    if metrics.latest_close is None or metrics.return_total is None:
        limitation = "Stored price history does not contain enough closes for return analysis."
        return _limited_finding(
            StockPriceAnalysisSection.RECENT_PERFORMANCE,
            "Recent Performance",
            limitation,
        )
    return StockPriceFinding(
        id="finding:recent_performance",
        section=StockPriceAnalysisSection.RECENT_PERFORMANCE,
        title="Recent Performance",
        summary=(
            f"Latest stored close is {_format_decimal(metrics.latest_close)} and total "
            f"stored-period return is {_format_ratio(metrics.return_total)}."
        ),
        confidence=ConfidenceLabel.HIGH,
        metric_names=("latest_close", "return_total", "return_1d"),
    )


def _trend_finding(metrics: MarketDataMetrics) -> StockPriceFinding:
    trend = _trend(metrics)
    if trend == StockTrendDirection.UNAVAILABLE:
        limitation = "Stored price history does not contain enough metrics to classify trend."
        return StockPriceFinding(
            id="finding:trend",
            section=StockPriceAnalysisSection.TREND,
            title="Trend",
            summary=limitation,
            confidence=ConfidenceLabel.UNKNOWN,
            trend=trend,
            limitations=(limitation,),
        )
    warnings = ()
    if 20 not in metrics.moving_averages:
        warnings = ("20-bar moving average is unavailable; trend uses shorter evidence.",)
    return StockPriceFinding(
        id="finding:trend",
        section=StockPriceAnalysisSection.TREND,
        title="Trend",
        summary=f"Deterministic trend classification is {trend.value}.",
        confidence=ConfidenceLabel.MEDIUM,
        metric_names=("return_total", "moving_average_5", "moving_average_20"),
        trend=trend,
        warnings=warnings,
    )


def _volatility_finding(metrics: MarketDataMetrics) -> StockPriceFinding:
    if metrics.volatility is None:
        limitation = "Stored price history does not contain enough returns for volatility."
        return _limited_finding(StockPriceAnalysisSection.VOLATILITY, "Volatility", limitation)
    return StockPriceFinding(
        id="finding:volatility",
        section=StockPriceAnalysisSection.VOLATILITY,
        title="Volatility",
        summary=f"Annualized volatility is {_format_ratio(metrics.volatility)}.",
        confidence=ConfidenceLabel.HIGH,
        metric_names=("annualized_volatility",),
    )


def _drawdown_finding(metrics: MarketDataMetrics) -> StockPriceFinding:
    if metrics.max_drawdown is None:
        limitation = "Stored price history does not contain enough closes for drawdown."
        return _limited_finding(StockPriceAnalysisSection.DRAWDOWN, "Drawdown", limitation)
    return StockPriceFinding(
        id="finding:drawdown",
        section=StockPriceAnalysisSection.DRAWDOWN,
        title="Drawdown",
        summary=f"Maximum stored-period drawdown is {_format_ratio(metrics.max_drawdown)}.",
        confidence=ConfidenceLabel.HIGH,
        metric_names=("max_drawdown",),
    )


def _volume_finding(bars: tuple[HistoricalPriceBar, ...]) -> StockPriceFinding:
    latest = bars[-1] if bars else None
    average = _average_volume(bars)
    if latest is None or average is None:
        limitation = "Stored price history does not contain volume observations."
        return _limited_finding(StockPriceAnalysisSection.VOLUME, "Volume", limitation)
    summary = (
        f"Latest stored volume is {latest.volume} and average stored volume is "
        f"{_format_decimal(average)}."
    )
    return StockPriceFinding(
        id="finding:volume",
        section=StockPriceAnalysisSection.VOLUME,
        title="Volume",
        summary=summary,
        confidence=ConfidenceLabel.HIGH,
        metric_names=("latest_volume", "average_volume", "latest_volume_vs_average"),
    )


def _benchmark_finding(
    metrics: MarketDataMetrics,
    benchmark_metrics: MarketDataMetrics,
) -> StockPriceFinding:
    if metrics.return_total is None or benchmark_metrics.return_total is None:
        limitation = "Stored data is insufficient for benchmark return comparison."
        return _limited_finding(
            StockPriceAnalysisSection.BENCHMARK_COMPARISON,
            "Benchmark Comparison",
            limitation,
        )
    relative = metrics.return_total - benchmark_metrics.return_total
    return StockPriceFinding(
        id="finding:benchmark_comparison",
        section=StockPriceAnalysisSection.BENCHMARK_COMPARISON,
        title="Benchmark Comparison",
        summary=(f"Stored-period relative return versus benchmark is {_format_ratio(relative)}."),
        confidence=ConfidenceLabel.MEDIUM,
        metric_names=("benchmark_return_total", "relative_return_vs_benchmark"),
    )


def _limited_finding(
    section: StockPriceAnalysisSection,
    title: str,
    limitation: str,
) -> StockPriceFinding:
    return StockPriceFinding(
        id=f"finding:{section.value}",
        section=section,
        title=title,
        summary=limitation,
        confidence=ConfidenceLabel.UNKNOWN,
        limitations=(limitation,),
    )


def _analysis_status(
    *,
    findings: tuple[StockPriceFinding, ...],
    limitations: list[str],
) -> StockPriceAnalysisStatus:
    if limitations or any(finding.limitations for finding in findings):
        return StockPriceAnalysisStatus.PARTIAL
    return StockPriceAnalysisStatus.COMPLETE


def _trend(metrics: MarketDataMetrics) -> StockTrendDirection:
    if metrics.latest_close is None or metrics.return_total is None:
        return StockTrendDirection.UNAVAILABLE
    moving_average = metrics.moving_averages.get(20) or metrics.moving_averages.get(5)
    if moving_average is None:
        if abs(metrics.return_total) < Decimal("0.02"):
            return StockTrendDirection.FLAT
        return StockTrendDirection.UP if metrics.return_total > 0 else StockTrendDirection.DOWN
    if abs(metrics.return_total) < Decimal("0.02"):
        return StockTrendDirection.FLAT
    if metrics.return_total > 0 and metrics.latest_close >= moving_average:
        return StockTrendDirection.UP
    if metrics.return_total < 0 and metrics.latest_close <= moving_average:
        return StockTrendDirection.DOWN
    return StockTrendDirection.MIXED


def _chart_series(
    series_id: str,
    label: str,
    bars: tuple[HistoricalPriceBar, ...],
) -> StockChartSeries | None:
    if not bars:
        return None
    return StockChartSeries(
        id=series_id,
        label=label,
        symbol=bars[-1].security.symbol,
        currency=bars[-1].security.currency,
        points=tuple(_chart_point(bars, index) for index, _ in enumerate(bars)),
    )


def _chart_point(bars: tuple[HistoricalPriceBar, ...], index: int) -> StockChartPoint:
    bar = bars[index]
    return StockChartPoint(
        priced_at=bar.priced_at,
        close=bar.close,
        adjusted_close=bar.adjusted_close,
        volume=bar.volume,
        moving_averages=_moving_average_points(bars[: index + 1]),
    )


def _moving_average_points(bars: tuple[HistoricalPriceBar, ...]) -> dict[int, Decimal]:
    closes = tuple(bar.close for bar in bars)
    return {
        window: sum(closes[-window:], Decimal("0")) / Decimal(window)
        for window in (5, 20)
        if len(closes) >= window
    }


def _sorted_bars(bars: tuple[HistoricalPriceBar, ...]) -> tuple[HistoricalPriceBar, ...]:
    return tuple(sorted(bars, key=lambda bar: bar.priced_at))


def _window_return(bars: tuple[HistoricalPriceBar, ...], window: int) -> Decimal | None:
    if len(bars) <= window:
        return None
    previous = bars[-window - 1].close
    latest = bars[-1].close
    if previous == 0:
        return None
    try:
        return (latest - previous) / previous
    except InvalidOperation, ZeroDivisionError:
        return None


def _average_volume(bars: tuple[HistoricalPriceBar, ...]) -> Decimal | None:
    if not bars:
        return None
    return sum((Decimal(bar.volume) for bar in bars), Decimal("0")) / Decimal(len(bars))


def _source_warnings(history: HistoricalPriceResult) -> tuple[str, ...]:
    warnings = list(history.warnings)
    if history.source.freshness_warning is not None:
        warnings.append(history.source.freshness_warning)
    if _has_adjusted_close(history.bars):
        warnings.append(SPLIT_ADJUSTMENT_WARNING)
    if history.security.currency is None:
        warnings.append("Currency metadata is unavailable for this market data series.")
    return tuple(dict.fromkeys(warnings))


def _has_adjusted_close(bars: tuple[HistoricalPriceBar, ...]) -> bool:
    return any(bar.adjusted_close is not None for bar in bars)


def _security_from_history(
    history: HistoricalPriceResult,
    *,
    fallback: StockPriceAnalysisSecurity | None = None,
) -> StockPriceAnalysisSecurity:
    return StockPriceAnalysisSecurity(
        symbol=history.security.symbol,
        security_id=history.security.security_id or (fallback.security_id if fallback else None),
        exchange_mic=history.security.exchange_mic or (fallback.exchange_mic if fallback else None),
        exchange_name=(
            history.security.exchange_name or (fallback.exchange_name if fallback else None)
        ),
        currency=history.security.currency or (fallback.currency if fallback else None),
    )


def _metric(
    name: str,
    value: Decimal,
    unit: str,
    description: str,
) -> StockPriceMetric:
    return StockPriceMetric(name=name, value=value, unit=unit, description=description)


def _format_ratio(value: Decimal) -> str:
    return f"{(value * Decimal('100')).quantize(Decimal('0.1'))}%"


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return format(value.quantize(Decimal("1")), "f")
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def _aware_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
