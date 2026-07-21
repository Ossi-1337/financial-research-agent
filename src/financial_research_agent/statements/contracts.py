from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from financial_research_agent.domain import FinancialStatementType


class FinancialStatementProviderName(StrEnum):
    SEC_COMPANY_FACTS = "sec-companyfacts"


class FinancialStatementPeriodType(StrEnum):
    ANNUAL = "annual"


class FinancialStatementErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class FinancialStatementCompany:
    cik: str
    company_id: str | None = None
    legal_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cik", _normalize_cik(self.cik))
        object.__setattr__(self, "company_id", _optional_text(self.company_id))
        object.__setattr__(self, "legal_name", _optional_text(self.legal_name))

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)

    def to_dict(self) -> dict[str, object]:
        return {
            "cik": self.cik,
            "padded_cik": self.padded_cik,
            "company_id": self.company_id,
            "legal_name": self.legal_name,
        }


@dataclass(frozen=True, slots=True)
class FinancialStatementPeriod:
    fiscal_year: int
    fiscal_period: str
    period_type: FinancialStatementPeriodType
    period_start: date | None
    period_end: date
    form: str | None = None
    accession_number: str | None = None
    filed_at: date | None = None

    def __post_init__(self) -> None:
        if self.fiscal_year <= 0:
            raise ValueError("fiscal_year must be positive")
        object.__setattr__(
            self,
            "fiscal_period",
            _require_text("fiscal_period", self.fiscal_period),
        )
        object.__setattr__(self, "period_type", FinancialStatementPeriodType(self.period_type))
        object.__setattr__(self, "form", _optional_text(self.form))
        object.__setattr__(self, "accession_number", _optional_text(self.accession_number))
        if self.period_start is not None and self.period_start > self.period_end:
            raise ValueError("period_start must be before or equal to period_end")

    def to_dict(self) -> dict[str, object]:
        return {
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "period_type": self.period_type.value,
            "period_start": (
                self.period_start.isoformat() if self.period_start is not None else None
            ),
            "period_end": self.period_end.isoformat(),
            "form": self.form,
            "accession_number": self.accession_number,
            "filed_at": self.filed_at.isoformat() if self.filed_at is not None else None,
        }


@dataclass(frozen=True, slots=True)
class FinancialStatementSource:
    provider: str
    provider_status: str
    source_url: str
    retrieved_at: datetime
    attribution: str
    data_as_of: date | None = None
    freshness_warning: str | None = None
    taxonomy_namespaces: tuple[str, ...] = ()
    concept_mappings: Mapping[str, str] = field(default_factory=dict)

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
        object.__setattr__(
            self,
            "taxonomy_namespaces",
            _text_tuple("taxonomy_namespaces", self.taxonomy_namespaces),
        )
        object.__setattr__(
            self,
            "concept_mappings",
            _text_mapping("concept_mappings", self.concept_mappings),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "provider_status": self.provider_status,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of is not None else None,
            "attribution": self.attribution,
            "freshness_warning": self.freshness_warning,
            "taxonomy_namespaces": list(self.taxonomy_namespaces),
            "concept_mappings": dict(self.concept_mappings),
        }


@dataclass(frozen=True, slots=True)
class NormalizedFinancialStatement:
    id: str
    company: FinancialStatementCompany
    statement_type: FinancialStatementType
    period: FinancialStatementPeriod
    currency: str
    line_items: Mapping[str, Decimal]
    source: FinancialStatementSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.company, FinancialStatementCompany):
            raise ValueError("company must be a FinancialStatementCompany")
        object.__setattr__(self, "statement_type", FinancialStatementType(self.statement_type))
        if not isinstance(self.period, FinancialStatementPeriod):
            raise ValueError("period must be a FinancialStatementPeriod")
        object.__setattr__(self, "currency", _require_text("currency", self.currency).upper())
        line_items = _decimal_mapping("line_items", self.line_items)
        if not line_items:
            raise ValueError("line_items must contain at least one item")
        object.__setattr__(self, "line_items", line_items)
        if not isinstance(self.source, FinancialStatementSource):
            raise ValueError("source must be a FinancialStatementSource")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "company": self.company.to_dict(),
            "statement_type": self.statement_type.value,
            "period": self.period.to_dict(),
            "currency": self.currency,
            "line_items": {key: str(value) for key, value in self.line_items.items()},
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FinancialStatementResult:
    company: FinancialStatementCompany
    statements: tuple[NormalizedFinancialStatement, ...]
    source: FinancialStatementSource
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.company, FinancialStatementCompany):
            raise ValueError("company must be a FinancialStatementCompany")
        statements = tuple(self.statements)
        for index, statement in enumerate(statements):
            if not isinstance(statement, NormalizedFinancialStatement):
                raise ValueError(f"statements[{index}] must be a NormalizedFinancialStatement")
        if not isinstance(self.source, FinancialStatementSource):
            raise ValueError("source must be a FinancialStatementSource")
        object.__setattr__(self, "statements", statements)
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "company": self.company.to_dict(),
            "statements": [statement.to_dict() for statement in self.statements],
            "source": self.source.to_dict(),
            "warnings": list(self.warnings),
        }


class FinancialStatementProvider(Protocol):
    async def fetch_statements(
        self,
        company: FinancialStatementCompany,
        *,
        fiscal_years: int = 3,
    ) -> FinancialStatementResult: ...


class FinancialStatementError(Exception):
    def __init__(
        self,
        *,
        code: FinancialStatementErrorCode | str,
        message: str,
        provider: str,
        retryable: bool = False,
    ) -> None:
        self.code = FinancialStatementErrorCode(code)
        self.message = _require_text("message", message)
        self.provider = _require_text("provider", provider)
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "financial_statement_error",
            "code": self.code.value,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
        }


def _normalize_cik(value: str) -> str:
    text = _require_text("cik", value)
    if text.upper().startswith("CIK"):
        text = text[3:]
    digits = text.lstrip("0") or "0"
    if not digits.isdigit() or int(digits) <= 0:
        raise ValueError("cik must contain positive digits")
    if len(digits) > 10:
        raise ValueError("cik must be at most 10 digits")
    return digits


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


def _decimal_mapping(name: str, values: Mapping[str, object]) -> Mapping[str, Decimal]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _decimal(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _decimal(name: str, value: object) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a decimal value") from exc
