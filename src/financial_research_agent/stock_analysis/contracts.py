from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from financial_research_agent.market_data import MarketDataSource

NO_TRADING_SIGNAL_NOTICE = (
    "This stock price analysis is market data research only and does not provide trading "
    "signals, price targets, or buy/sell/hold recommendations."
)


class StockPriceAnalysisStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_DATA = "no_data"


class StockPriceAnalysisSection(StrEnum):
    RECENT_PERFORMANCE = "recent_performance"
    TREND = "trend"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    VOLUME = "volume"
    BENCHMARK_COMPARISON = "benchmark_comparison"


class StockTrendDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StockPriceAnalysisSecurity:
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
class StockPriceMetric:
    name: str
    value: Decimal
    unit: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "value", _decimal("value", self.value))
        object.__setattr__(self, "unit", _require_text("unit", self.unit))
        object.__setattr__(self, "description", _require_text("description", self.description))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": _decimal_string(self.value),
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class StockPriceFinding:
    id: str
    section: StockPriceAnalysisSection
    title: str
    summary: str
    confidence: ConfidenceLabel
    metric_names: tuple[str, ...] = ()
    trend: StockTrendDirection | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "section", StockPriceAnalysisSection(self.section))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "confidence", ConfidenceLabel(self.confidence))
        object.__setattr__(self, "metric_names", _text_tuple("metric_names", self.metric_names))
        object.__setattr__(
            self,
            "trend",
            StockTrendDirection(self.trend) if self.trend is not None else None,
        )
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        if not self.metric_names and not self.limitations:
            raise ValueError("finding must include metric_names or limitations")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "section": self.section.value,
            "title": self.title,
            "summary": self.summary,
            "confidence": self.confidence.value,
            "metric_names": list(self.metric_names),
            "trend": self.trend.value if self.trend is not None else None,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class StockChartPoint:
    priced_at: date
    close: Decimal
    volume: int
    adjusted_close: Decimal | None = None
    moving_averages: Mapping[int, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "close", _non_negative_decimal("close", self.close))
        if self.volume < 0:
            raise ValueError("volume must be non-negative")
        if self.adjusted_close is not None:
            object.__setattr__(
                self,
                "adjusted_close",
                _non_negative_decimal("adjusted_close", self.adjusted_close),
            )
        object.__setattr__(self, "moving_averages", _int_decimal_mapping(self.moving_averages))

    def to_dict(self) -> dict[str, object]:
        return {
            "priced_at": self.priced_at.isoformat(),
            "close": _decimal_string(self.close),
            "adjusted_close": _optional_decimal_string(self.adjusted_close),
            "volume": self.volume,
            "moving_averages": {
                str(window): _decimal_string(value)
                for window, value in self.moving_averages.items()
            },
        }


@dataclass(frozen=True, slots=True)
class StockChartSeries:
    id: str
    label: str
    symbol: str
    currency: str | None
    points: tuple[StockChartPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "label", _require_text("label", self.label))
        object.__setattr__(self, "symbol", _require_text("symbol", self.symbol).upper())
        object.__setattr__(self, "currency", _optional_upper_text(self.currency))
        points = tuple(self.points)
        for index, point in enumerate(points):
            if not isinstance(point, StockChartPoint):
                raise ValueError(f"points[{index}] must be a StockChartPoint")
        object.__setattr__(self, "points", points)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "symbol": self.symbol,
            "currency": self.currency,
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class StockPriceAnalysisResult:
    id: str
    security: StockPriceAnalysisSecurity
    status: StockPriceAnalysisStatus
    created_at: datetime
    metrics: tuple[StockPriceMetric, ...]
    findings: tuple[StockPriceFinding, ...]
    chart_series: tuple[StockChartSeries, ...]
    primary_source: MarketDataSource | None = None
    benchmark_security: StockPriceAnalysisSecurity | None = None
    benchmark_source: MarketDataSource | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    no_trading_signal_notice: str = NO_TRADING_SIGNAL_NOTICE

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.security, StockPriceAnalysisSecurity):
            raise ValueError("security must be a StockPriceAnalysisSecurity")
        object.__setattr__(self, "status", StockPriceAnalysisStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "metrics", _metric_tuple(self.metrics))
        object.__setattr__(self, "findings", _finding_tuple(self.findings))
        object.__setattr__(self, "chart_series", _series_tuple(self.chart_series))
        if self.primary_source is not None and not isinstance(
            self.primary_source,
            MarketDataSource,
        ):
            raise ValueError("primary_source must be a MarketDataSource")
        if self.benchmark_security is not None and not isinstance(
            self.benchmark_security, StockPriceAnalysisSecurity
        ):
            raise ValueError("benchmark_security must be a StockPriceAnalysisSecurity")
        if self.benchmark_source is not None and not isinstance(
            self.benchmark_source, MarketDataSource
        ):
            raise ValueError("benchmark_source must be a MarketDataSource")
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(
            self,
            "no_trading_signal_notice",
            _require_text("no_trading_signal_notice", self.no_trading_signal_notice),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "security": self.security.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "findings": [finding.to_dict() for finding in self.findings],
            "chart_series": [series.to_dict() for series in self.chart_series],
            "primary_source": (
                self.primary_source.to_dict() if self.primary_source is not None else None
            ),
            "benchmark_security": (
                self.benchmark_security.to_dict() if self.benchmark_security is not None else None
            ),
            "benchmark_source": (
                self.benchmark_source.to_dict() if self.benchmark_source is not None else None
            ),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "no_trading_signal_notice": self.no_trading_signal_notice,
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


def _metric_tuple(values: Iterable[StockPriceMetric]) -> tuple[StockPriceMetric, ...]:
    metrics = tuple(values)
    for index, metric in enumerate(metrics):
        if not isinstance(metric, StockPriceMetric):
            raise ValueError(f"metrics[{index}] must be a StockPriceMetric")
    return metrics


def _finding_tuple(values: Iterable[StockPriceFinding]) -> tuple[StockPriceFinding, ...]:
    findings = tuple(values)
    for index, finding in enumerate(findings):
        if not isinstance(finding, StockPriceFinding):
            raise ValueError(f"findings[{index}] must be a StockPriceFinding")
    return findings


def _series_tuple(values: Iterable[StockChartSeries]) -> tuple[StockChartSeries, ...]:
    series = tuple(values)
    for index, item in enumerate(series):
        if not isinstance(item, StockChartSeries):
            raise ValueError(f"chart_series[{index}] must be a StockChartSeries")
    return series


def _int_decimal_mapping(values: Mapping[int, Decimal]) -> Mapping[int, Decimal]:
    if not isinstance(values, Mapping):
        raise ValueError("moving_averages must be a mapping")
    return MappingProxyType(
        {
            int(window): _non_negative_decimal(f"moving_averages[{window}]", value)
            for window, value in values.items()
        }
    )


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


def _decimal_string(value: Decimal) -> str:
    if value == value.to_integral():
        return format(value.quantize(Decimal("1")), "f")
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def _optional_decimal_string(value: Decimal | None) -> str | None:
    return _decimal_string(value) if value is not None else None
