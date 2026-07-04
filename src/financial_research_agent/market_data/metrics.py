from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

from financial_research_agent.market_data.contracts import (
    HistoricalPriceBar,
    MarketDataMetrics,
)

DEFAULT_MOVING_AVERAGE_WINDOWS = (5, 20)
TRADING_DAYS_PER_YEAR = Decimal("252")


def calculate_price_metrics(
    bars: tuple[HistoricalPriceBar, ...],
    *,
    moving_average_windows: tuple[int, ...] = DEFAULT_MOVING_AVERAGE_WINDOWS,
) -> MarketDataMetrics:
    if not bars:
        return MarketDataMetrics(
            symbol="unknown",
            latest_close=None,
            return_1d=None,
            return_total=None,
            moving_averages={},
            volatility=None,
            max_drawdown=None,
        )
    sorted_bars = tuple(sorted(bars, key=lambda bar: bar.priced_at))
    symbol = sorted_bars[-1].security.symbol
    closes = tuple(bar.close for bar in sorted_bars)
    returns = _daily_returns(closes)
    moving_averages = {
        window: _average(closes[-window:])
        for window in moving_average_windows
        if window > 0 and len(closes) >= window
    }
    return MarketDataMetrics(
        symbol=symbol,
        latest_close=closes[-1],
        return_1d=returns[-1] if returns else None,
        return_total=_relative_change(closes[0], closes[-1]) if len(closes) > 1 else None,
        moving_averages=moving_averages,
        volatility=_annualized_volatility(returns) if len(returns) > 1 else None,
        max_drawdown=_max_drawdown(closes),
    )


def _daily_returns(closes: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for previous, current in pairwise(closes):
        if previous != 0:
            values.append(_relative_change(previous, current))
    return tuple(values)


def _relative_change(previous: Decimal, current: Decimal) -> Decimal:
    return (current - previous) / previous


def _average(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def _annualized_volatility(returns: tuple[Decimal, ...]) -> Decimal:
    mean = _average(returns)
    variance = sum((value - mean) ** 2 for value in returns) / Decimal(len(returns))
    return variance.sqrt() * TRADING_DAYS_PER_YEAR.sqrt()


def _max_drawdown(closes: tuple[Decimal, ...]) -> Decimal | None:
    if not closes:
        return None
    peak = closes[0]
    drawdown = Decimal("0")
    for close in closes:
        if close > peak:
            peak = close
        if peak != 0:
            current_drawdown = (close - peak) / peak
            if current_drawdown < drawdown:
                drawdown = current_drawdown
    return drawdown
