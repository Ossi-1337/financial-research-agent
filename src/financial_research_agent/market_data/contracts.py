from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class MarketDataProviderName(StrEnum):
    ALPHA_VANTAGE = "alpha-vantage"


class MarketDataErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class MarketDataSource:
    provider: str
    provider_status: str
    source_url: str
    retrieved_at: datetime
    attribution: str
    data_as_of: date | None = None
    freshness_warning: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(
            self,
            "provider_status",
            _require_text("provider_status", self.provider_status),
        )
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(self, "attribution", _require_text("attribution", self.attribution))
        object.__setattr__(self, "freshness_warning", _optional_text(self.freshness_warning))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_status": self.provider_status,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of is not None else None,
            "attribution": self.attribution,
            "freshness_warning": self.freshness_warning,
        }


@dataclass(frozen=True, slots=True)
class MarketSecurity:
    symbol: str
    security_id: str | None = None
    exchange_mic: str | None = None
    exchange_name: str | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_text("symbol", self.symbol).upper())
        object.__setattr__(self, "security_id", _optional_text(self.security_id))
        object.__setattr__(self, "exchange_mic", _optional_upper_text(self.exchange_mic))
        object.__setattr__(self, "exchange_name", _optional_text(self.exchange_name))
        object.__setattr__(self, "currency", _optional_upper_text(self.currency))

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_mic": self.exchange_mic,
            "exchange_name": self.exchange_name,
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class HistoricalPriceBar:
    security: MarketSecurity
    priced_at: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.security, MarketSecurity):
            raise ValueError("security must be a MarketSecurity")
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _non_negative_decimal(name, getattr(self, name)))
        if self.adjusted_close is not None:
            object.__setattr__(
                self,
                "adjusted_close",
                _non_negative_decimal("adjusted_close", self.adjusted_close),
            )
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

    @property
    def id(self) -> str:
        return f"{self.security.symbol}:{self.priced_at.isoformat()}"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "security": self.security.to_dict(),
            "priced_at": self.priced_at.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": self.volume,
            "adjusted_close": str(self.adjusted_close) if self.adjusted_close is not None else None,
        }


@dataclass(frozen=True, slots=True)
class MarketQuote:
    security: MarketSecurity
    price: Decimal
    volume: int | None
    trading_day: date | None
    source: MarketDataSource

    def __post_init__(self) -> None:
        if not isinstance(self.security, MarketSecurity):
            raise ValueError("security must be a MarketSecurity")
        object.__setattr__(self, "price", _non_negative_decimal("price", self.price))
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume must be non-negative")
        if not isinstance(self.source, MarketDataSource):
            raise ValueError("source must be a MarketDataSource")

    def to_dict(self) -> dict[str, object]:
        return {
            "security": self.security.to_dict(),
            "price": str(self.price),
            "volume": self.volume,
            "trading_day": self.trading_day.isoformat() if self.trading_day is not None else None,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MarketDataMetrics:
    symbol: str
    latest_close: Decimal | None
    return_1d: Decimal | None
    return_total: Decimal | None
    moving_averages: Mapping[int, Decimal]
    volatility: Decimal | None
    max_drawdown: Decimal | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_text("symbol", self.symbol).upper())
        if self.latest_close is not None:
            object.__setattr__(
                self, "latest_close", _non_negative_decimal("latest_close", self.latest_close)
            )
        if self.return_1d is not None:
            object.__setattr__(self, "return_1d", _decimal("return_1d", self.return_1d))
        if self.return_total is not None:
            object.__setattr__(self, "return_total", _decimal("return_total", self.return_total))
        object.__setattr__(
            self,
            "moving_averages",
            MappingProxyType(
                {
                    int(window): _non_negative_decimal(f"moving_averages[{window}]", value)
                    for window, value in self.moving_averages.items()
                }
            ),
        )
        if self.volatility is not None:
            object.__setattr__(
                self, "volatility", _non_negative_decimal("volatility", self.volatility)
            )
        if self.max_drawdown is not None:
            object.__setattr__(self, "max_drawdown", _decimal("max_drawdown", self.max_drawdown))

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "latest_close": _decimal_string(self.latest_close),
            "return_1d": _decimal_string(self.return_1d),
            "return_total": _decimal_string(self.return_total),
            "moving_averages": {
                str(window): _decimal_string(value)
                for window, value in self.moving_averages.items()
            },
            "volatility": _decimal_string(self.volatility),
            "max_drawdown": _decimal_string(self.max_drawdown),
        }


@dataclass(frozen=True, slots=True)
class HistoricalPriceResult:
    security: MarketSecurity
    bars: tuple[HistoricalPriceBar, ...]
    source: MarketDataSource
    metrics: MarketDataMetrics
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.security, MarketSecurity):
            raise ValueError("security must be a MarketSecurity")
        bars = tuple(self.bars)
        for index, bar in enumerate(bars):
            if not isinstance(bar, HistoricalPriceBar):
                raise ValueError(f"bars[{index}] must be a HistoricalPriceBar")
        if not isinstance(self.source, MarketDataSource):
            raise ValueError("source must be a MarketDataSource")
        if not isinstance(self.metrics, MarketDataMetrics):
            raise ValueError("metrics must be MarketDataMetrics")
        object.__setattr__(self, "bars", bars)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "security": self.security.to_dict(),
            "bars": [bar.to_dict() for bar in self.bars],
            "source": self.source.to_dict(),
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
        }


class MarketDataProvider(Protocol):
    async def fetch_daily_prices(
        self,
        security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult: ...

    async def fetch_quote(self, security: MarketSecurity) -> MarketQuote: ...


class MarketDataError(Exception):
    def __init__(
        self,
        *,
        code: MarketDataErrorCode | str,
        message: str,
        provider: str,
        retryable: bool = False,
    ) -> None:
        self.code = MarketDataErrorCode(code)
        self.message = _require_text("message", message)
        self.provider = _require_text("provider", provider)
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "market_data_error",
            "code": self.code.value,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _optional_upper_text(value: str | None) -> str | None:
    text = _optional_text(value)
    return text.upper() if text is not None else None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _decimal(name: str, value: object) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a decimal value") from exc


def _non_negative_decimal(name: str, value: object) -> Decimal:
    amount = _decimal(name, value)
    if amount < 0:
        raise ValueError(f"{name} must be non-negative")
    return amount


def _decimal_string(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
