from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, Self

from financial_research_agent.market_data.contracts import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataMetrics,
    MarketDataSource,
    MarketSecurity,
)
from financial_research_agent.market_data.metrics import calculate_price_metrics
from financial_research_agent.settings import Settings

MARKET_DATA_STORE_VERSION = 1


class MarketDataStore:
    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        stale_after: timedelta = timedelta(days=1),
    ) -> None:
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self.storage_path = storage_path
        self.stale_after = stale_after
        self._series: dict[str, HistoricalPriceResult] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            storage_path=settings.local_paths.data_dir / "market_data_price_bars.json",
            stale_after=timedelta(days=settings.data_sources.market_data_cache_ttl_days),
        )

    def save_history(self, result: HistoricalPriceResult) -> HistoricalPriceResult:
        key = _series_key(result.source.provider, result.security.symbol)
        with self._lock:
            self._series[key] = result
            self._save()
        return result

    def get_history(
        self,
        *,
        symbol: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> HistoricalPriceResult | None:
        normalized_symbol = _require_text("symbol", symbol).upper()
        with self._lock:
            candidates = [
                result
                for result in self._series.values()
                if result.security.symbol == normalized_symbol
                and (provider is None or result.source.provider == provider)
            ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda result: result.source.retrieved_at)
        return self._with_freshness(latest, now or datetime.now(UTC))

    def count(self) -> int:
        with self._lock:
            return len(self._series)

    def list(self) -> tuple[HistoricalPriceResult, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._series.values(),
                    key=lambda result: result.source.retrieved_at,
                    reverse=True,
                )
            )

    def clear(self) -> int:
        with self._lock:
            deleted = len(self._series)
            self._series.clear()
            self._save()
            return deleted

    def _with_freshness(
        self, result: HistoricalPriceResult, now: datetime
    ) -> HistoricalPriceResult:
        if result.source.retrieved_at + self.stale_after > now:
            return result
        warning = "Stored market data is stale; refresh before relying on it."
        source = replace(result.source, freshness_warning=warning)
        warnings = tuple(dict.fromkeys((*result.warnings, warning)))
        return replace(result, source=source, warnings=warnings)

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != MARKET_DATA_STORE_VERSION:
                raise ValueError("unsupported market data store version")
            series_payload = payload.get("series", ())
            if not isinstance(series_payload, list):
                raise ValueError("market data series must be a list")
            self._series = {
                _series_key(result.source.provider, result.security.symbol): result
                for result in (historical_price_result_from_dict(item) for item in series_payload)
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load market data store: {self.storage_path}") from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MARKET_DATA_STORE_VERSION,
            "series": [result.to_dict() for result in self._series.values()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def historical_price_result_from_dict(payload: Any) -> HistoricalPriceResult:
    if not isinstance(payload, dict):
        raise ValueError("market data history must be an object")
    security = _security_from_payload(payload["security"])
    source = _source_from_payload(payload["source"])
    bars = tuple(_bar_from_payload(item, security) for item in payload.get("bars", ()))
    metrics = calculate_price_metrics(bars) if bars else _empty_metrics(security.symbol)
    warnings = tuple(str(item) for item in payload.get("warnings", ()))
    return HistoricalPriceResult(
        security=security,
        bars=bars,
        source=source,
        metrics=metrics,
        warnings=warnings,
    )


def _security_from_payload(payload: Any) -> MarketSecurity:
    if not isinstance(payload, dict):
        raise ValueError("security must be an object")
    return MarketSecurity(
        symbol=str(payload["symbol"]),
        security_id=_optional_payload_text(payload, "security_id"),
        exchange_mic=_optional_payload_text(payload, "exchange_mic"),
        exchange_name=_optional_payload_text(payload, "exchange_name"),
        currency=_optional_payload_text(payload, "currency"),
    )


def _source_from_payload(payload: Any) -> MarketDataSource:
    if not isinstance(payload, dict):
        raise ValueError("source must be an object")
    data_as_of = payload.get("data_as_of")
    return MarketDataSource(
        provider=str(payload["provider"]),
        provider_status=str(payload["provider_status"]),
        source_url=str(payload["source_url"]),
        retrieved_at=_datetime_from_payload(payload["retrieved_at"]),
        data_as_of=date.fromisoformat(data_as_of) if isinstance(data_as_of, str) else None,
        attribution=str(payload["attribution"]),
        freshness_warning=_optional_payload_text(payload, "freshness_warning"),
    )


def _bar_from_payload(payload: Any, security: MarketSecurity) -> HistoricalPriceBar:
    if not isinstance(payload, dict):
        raise ValueError("bar must be an object")
    adjusted_close = payload.get("adjusted_close")
    return HistoricalPriceBar(
        security=security,
        priced_at=date.fromisoformat(str(payload["priced_at"])),
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=int(payload["volume"]),
        adjusted_close=Decimal(str(adjusted_close)) if adjusted_close is not None else None,
    )


def _empty_metrics(symbol: str) -> MarketDataMetrics:
    return MarketDataMetrics(
        symbol=symbol,
        latest_close=None,
        return_1d=None,
        return_total=None,
        moving_averages={},
        volatility=None,
        max_drawdown=None,
    )


def _datetime_from_payload(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _optional_payload_text(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    return str(value)


def _series_key(provider: str, symbol: str) -> str:
    return f"{_require_text('provider', provider)}:{_require_text('symbol', symbol).upper()}"


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
