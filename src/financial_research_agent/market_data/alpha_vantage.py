from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import httpx

from financial_research_agent.market_data.contracts import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataError,
    MarketDataErrorCode,
    MarketDataProviderName,
    MarketDataSource,
    MarketQuote,
    MarketSecurity,
)
from financial_research_agent.market_data.metrics import calculate_price_metrics

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_ATTRIBUTION = "Alpha Vantage"
ALPHA_VANTAGE_PROVIDER_STATUS = "documented public API"
ALPHA_VANTAGE_PROVIDER = MarketDataProviderName.ALPHA_VANTAGE.value


@dataclass(frozen=True, slots=True)
class AlphaVantageProvider:
    api_key: str | None
    base_url: str = ALPHA_VANTAGE_BASE_URL
    http_client: httpx.AsyncClient | None = None
    now: Callable[[], datetime] | None = None
    minimum_request_interval_seconds: float = 2.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic
    _request_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _last_request_at: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds must not be negative")

    async def fetch_daily_prices(
        self,
        security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        self._require_api_key()
        normalized_outputsize = _outputsize(outputsize)
        payload = await self._get(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": security.symbol,
                "outputsize": normalized_outputsize,
                "apikey": self.api_key,
            }
        )
        time_series = payload.get("Time Series (Daily)")
        if not isinstance(time_series, Mapping):
            raise MarketDataError(
                code=MarketDataErrorCode.MALFORMED_RESPONSE,
                message="Alpha Vantage daily response did not include Time Series (Daily).",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        bars = tuple(
            sorted(
                (
                    _daily_bar_from_payload(security, day, values)
                    for day, values in time_series.items()
                ),
                key=lambda bar: bar.priced_at,
            )
        )
        if not bars:
            raise MarketDataError(
                code=MarketDataErrorCode.NOT_FOUND,
                message=f"No daily price bars returned for symbol {security.symbol}.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        source = self._source(data_as_of=bars[-1].priced_at)
        warnings = _metadata_warnings(security)
        return HistoricalPriceResult(
            security=security,
            bars=bars,
            source=source,
            metrics=calculate_price_metrics(bars),
            warnings=warnings,
        )

    async def fetch_quote(self, security: MarketSecurity) -> MarketQuote:
        self._require_api_key()
        payload = await self._get(
            {
                "function": "GLOBAL_QUOTE",
                "symbol": security.symbol,
                "apikey": self.api_key,
            }
        )
        quote = payload.get("Global Quote")
        if not isinstance(quote, Mapping) or not quote:
            raise MarketDataError(
                code=MarketDataErrorCode.NOT_FOUND,
                message=f"No quote returned for symbol {security.symbol}.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        trading_day = str(quote.get("07. latest trading day", "")).strip()
        return MarketQuote(
            security=security,
            price=Decimal(str(quote["05. price"])),
            volume=_optional_int(quote.get("06. volume")),
            trading_day=date.fromisoformat(trading_day) if trading_day else None,
            source=self._source(
                data_as_of=date.fromisoformat(trading_day) if trading_day else None,
            ),
        )

    async def _get(self, params: Mapping[str, object]) -> Mapping[str, Any]:
        await self._wait_for_request_slot()
        try:
            if self.http_client is None:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(self.base_url, params=params)
            else:
                response = await self.http_client.get(self.base_url, params=params)
        except httpx.TimeoutException as exc:
            raise MarketDataError(
                code=MarketDataErrorCode.TIMEOUT,
                message="Timed out while fetching Alpha Vantage market data.",
                provider=ALPHA_VANTAGE_PROVIDER,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise MarketDataError(
                code=MarketDataErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Alpha Vantage market data source is unavailable: {exc}",
                provider=ALPHA_VANTAGE_PROVIDER,
                retryable=True,
            ) from exc
        if response.status_code == 401 or response.status_code == 403:
            raise MarketDataError(
                code=MarketDataErrorCode.AUTHENTICATION_FAILED,
                message="Alpha Vantage authentication failed.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        if response.status_code == 429:
            raise MarketDataError(
                code=MarketDataErrorCode.RATE_LIMITED,
                message="Alpha Vantage rate limited the request.",
                provider=ALPHA_VANTAGE_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 500:
            raise MarketDataError(
                code=MarketDataErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Alpha Vantage returned HTTP {response.status_code}.",
                provider=ALPHA_VANTAGE_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 400:
            raise MarketDataError(
                code=MarketDataErrorCode.INVALID_REQUEST,
                message=f"Alpha Vantage returned HTTP {response.status_code}.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketDataError(
                code=MarketDataErrorCode.MALFORMED_RESPONSE,
                message="Alpha Vantage returned malformed JSON.",
                provider=ALPHA_VANTAGE_PROVIDER,
            ) from exc
        if not isinstance(payload, Mapping):
            raise MarketDataError(
                code=MarketDataErrorCode.MALFORMED_RESPONSE,
                message="Alpha Vantage response must be a JSON object.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )
        _raise_for_alpha_vantage_message(payload)
        return payload

    async def _wait_for_request_slot(self) -> None:
        async with self._request_lock:
            now = self.monotonic()
            if self._last_request_at:
                remaining = self.minimum_request_interval_seconds - (now - self._last_request_at[0])
                if remaining > 0:
                    await self.sleep(remaining)
                    now = self.monotonic()
            self._last_request_at[:] = [now]

    def _source(self, *, data_as_of: date | None = None) -> MarketDataSource:
        return MarketDataSource(
            provider=ALPHA_VANTAGE_PROVIDER,
            provider_status=ALPHA_VANTAGE_PROVIDER_STATUS,
            source_url=self.base_url,
            retrieved_at=(self.now or (lambda: datetime.now(UTC)))(),
            data_as_of=data_as_of,
            attribution=ALPHA_VANTAGE_ATTRIBUTION,
            freshness_warning=(
                "Alpha Vantage data may be delayed or provider-limited; verify before use."
            ),
        )

    def _require_api_key(self) -> None:
        if self.api_key is None or self.api_key.strip() == "":
            raise MarketDataError(
                code=MarketDataErrorCode.AUTHENTICATION_FAILED,
                message="FRA_ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY is required.",
                provider=ALPHA_VANTAGE_PROVIDER,
            )


def _daily_bar_from_payload(
    security: MarketSecurity,
    day: str,
    payload: Any,
) -> HistoricalPriceBar:
    if not isinstance(payload, Mapping):
        raise MarketDataError(
            code=MarketDataErrorCode.MALFORMED_RESPONSE,
            message="Alpha Vantage daily response contains a non-object bar.",
            provider=ALPHA_VANTAGE_PROVIDER,
        )
    try:
        return HistoricalPriceBar(
            security=security,
            priced_at=date.fromisoformat(day),
            open=Decimal(str(payload["1. open"])),
            high=Decimal(str(payload["2. high"])),
            low=Decimal(str(payload["3. low"])),
            close=Decimal(str(payload["4. close"])),
            volume=int(payload["5. volume"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketDataError(
            code=MarketDataErrorCode.MALFORMED_RESPONSE,
            message="Alpha Vantage daily response contains an invalid bar.",
            provider=ALPHA_VANTAGE_PROVIDER,
        ) from exc


def _raise_for_alpha_vantage_message(payload: Mapping[str, Any]) -> None:
    error_message = payload.get("Error Message")
    if isinstance(error_message, str) and error_message.strip():
        raise MarketDataError(
            code=MarketDataErrorCode.INVALID_REQUEST,
            message=error_message.strip(),
            provider=ALPHA_VANTAGE_PROVIDER,
        )
    note = payload.get("Note")
    information = payload.get("Information")
    message = (
        note if isinstance(note, str) else information if isinstance(information, str) else None
    )
    if message is not None and message.strip():
        raise MarketDataError(
            code=MarketDataErrorCode.RATE_LIMITED,
            message=message.strip(),
            provider=ALPHA_VANTAGE_PROVIDER,
            retryable=True,
        )


def _metadata_warnings(security: MarketSecurity) -> tuple[str, ...]:
    warnings: list[str] = []
    if security.currency is None:
        warnings.append("Currency metadata is unavailable for this security.")
    if security.exchange_mic is None and security.exchange_name is None:
        warnings.append("Exchange metadata is unavailable for this security.")
    return tuple(warnings)


def _outputsize(value: str) -> str:
    text = value.strip().lower()
    if text not in {"compact", "full"}:
        raise MarketDataError(
            code=MarketDataErrorCode.INVALID_REQUEST,
            message="outputsize must be compact or full",
            provider=ALPHA_VANTAGE_PROVIDER,
        )
    return text


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)
