from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self


class SecurityType(StrEnum):
    EQUITY = "equity"
    ADR = "adr"
    ETF = "etf"
    FUND = "fund"
    OTHER = "other"


class FilingType(StrEnum):
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    SEC_10K = "sec_10k"
    SEC_10Q = "sec_10q"
    COMPANY_REPORT = "company_report"
    OTHER = "other"


class FinancialStatementType(StrEnum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    KEY_RATIOS = "key_ratios"


class EvidenceSourceType(StrEnum):
    FILING = "filing"
    FINANCIAL_STATEMENT = "financial_statement"
    MARKET_DATA = "market_data"
    NEWS = "news"
    MACRO = "macro"
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResearchIssueCode(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    STALE_DATA = "stale_data"
    MISSING_TICKER = "missing_ticker"
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    PARTIAL_RESEARCH = "partial_research"


class ResearchRunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Company:
    id: str
    legal_name: str
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    country_code: str | None = None
    lei: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "legal_name", _require_text("legal_name", self.legal_name))
        object.__setattr__(self, "display_name", _optional_text(self.display_name))
        object.__setattr__(self, "aliases", _text_tuple("aliases", self.aliases))
        object.__setattr__(self, "country_code", _optional_upper_text(self.country_code))
        object.__setattr__(self, "lei", _optional_upper_text(self.lei))


@dataclass(frozen=True, slots=True)
class Exchange:
    id: str
    name: str
    mic: str
    country_code: str
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "mic", _require_text("mic", self.mic).upper())
        object.__setattr__(
            self, "country_code", _require_text("country_code", self.country_code).upper()
        )
        object.__setattr__(self, "currency", _require_text("currency", self.currency).upper())


@dataclass(frozen=True, slots=True)
class Security:
    id: str
    company_id: str
    exchange_id: str
    ticker: str
    name: str
    currency: str
    security_type: SecurityType = SecurityType.EQUITY
    isin: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "company_id", _require_text("company_id", self.company_id))
        object.__setattr__(self, "exchange_id", _require_text("exchange_id", self.exchange_id))
        object.__setattr__(self, "ticker", _require_text("ticker", self.ticker).upper())
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "currency", _require_text("currency", self.currency).upper())
        object.__setattr__(self, "isin", _optional_upper_text(self.isin))
        object.__setattr__(self, "security_type", SecurityType(self.security_type))


@dataclass(frozen=True, slots=True)
class Filing:
    id: str
    company_id: str
    filing_type: FilingType
    title: str
    period_end: date | None
    published_at: datetime | None
    source_url: str
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "company_id", _require_text("company_id", self.company_id))
        object.__setattr__(self, "filing_type", FilingType(self.filing_type))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))


@dataclass(frozen=True, slots=True)
class FinancialStatement:
    id: str
    company_id: str
    statement_type: FinancialStatementType
    period_start: date | None
    period_end: date
    fiscal_year: int
    currency: str
    line_items: Mapping[str, Decimal]
    source_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "company_id", _require_text("company_id", self.company_id))
        object.__setattr__(self, "statement_type", FinancialStatementType(self.statement_type))
        object.__setattr__(self, "currency", _require_text("currency", self.currency).upper())
        object.__setattr__(self, "line_items", _decimal_mapping("line_items", self.line_items))
        object.__setattr__(
            self,
            "source_evidence_ids",
            _text_tuple("source_evidence_ids", self.source_evidence_ids),
        )
        if self.fiscal_year <= 0:
            raise ValueError("fiscal_year must be positive")
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start must be before or equal to period_end")


@dataclass(frozen=True, slots=True)
class PriceBar:
    id: str
    security_id: str
    priced_at: date
    currency: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None
    adjusted_close: Decimal | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "security_id", _require_text("security_id", self.security_id))
        object.__setattr__(self, "currency", _require_text("currency", self.currency).upper())
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
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume must be non-negative")
        object.__setattr__(self, "source", _optional_text(self.source))


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    id: str
    source_type: EvidenceSourceType
    title: str
    retrieved_at: datetime
    source_url: str | None = None
    source_id: str | None = None
    excerpt: str | None = None
    location: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "source_type", EvidenceSourceType(self.source_type))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "source_url", _optional_text(self.source_url))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "excerpt", _optional_text(self.excerpt))
        object.__setattr__(self, "location", _optional_text(self.location))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))


@dataclass(frozen=True, slots=True)
class ResearchIssue:
    code: ResearchIssueCode
    message: str
    severity: IssueSeverity = IssueSeverity.WARNING
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ResearchIssueCode(self.code))
        object.__setattr__(self, "message", _require_text("message", self.message))
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(self, "source", _optional_text(self.source))

    @classmethod
    def provider_unavailable(cls, provider: str, message: str | None = None) -> Self:
        detail = message or f"Provider is unavailable: {provider}"
        return cls(
            code=ResearchIssueCode.PROVIDER_UNAVAILABLE,
            message=detail,
            severity=IssueSeverity.ERROR,
            source=provider,
        )


@dataclass(frozen=True, slots=True)
class AgentOutput:
    id: str
    agent_name: str
    created_at: datetime
    summary: str
    findings: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    issues: tuple[ResearchIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "agent_name", _require_text("agent_name", self.agent_name))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "findings", _text_tuple("findings", self.findings))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(self, "issues", _issue_tuple("issues", self.issues))


@dataclass(frozen=True, slots=True)
class ResearchRun:
    id: str
    query: str
    created_at: datetime
    status: ResearchRunStatus = ResearchRunStatus.CREATED
    company_id: str | None = None
    security_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    agent_output_ids: tuple[str, ...] = ()
    issues: tuple[ResearchIssue, ...] = ()
    final_answer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", ResearchRunStatus(self.status))
        object.__setattr__(self, "company_id", _optional_text(self.company_id))
        object.__setattr__(self, "security_ids", _text_tuple("security_ids", self.security_ids))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "agent_output_ids",
            _text_tuple("agent_output_ids", self.agent_output_ids),
        )
        object.__setattr__(self, "issues", _issue_tuple("issues", self.issues))
        object.__setattr__(self, "final_answer", _optional_text(self.final_answer))


def _require_text(name: str, value: str) -> str:
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


def _issue_tuple(name: str, values: Iterable[ResearchIssue]) -> tuple[ResearchIssue, ...]:
    issues = tuple(values)
    for index, issue in enumerate(issues):
        if not isinstance(issue, ResearchIssue):
            raise ValueError(f"{name}[{index}] must be a ResearchIssue")
    return issues


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _decimal_mapping(name: str, values: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _decimal(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _decimal(name: str, value: Any) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a decimal value") from exc


def _non_negative_decimal(name: str, value: Any) -> Decimal:
    amount = _decimal(name, value)
    if amount < 0:
        raise ValueError(f"{name} must be non-negative")
    return amount
