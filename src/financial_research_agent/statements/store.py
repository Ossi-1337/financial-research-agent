from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, Self

from financial_research_agent.domain import FinancialStatementType
from financial_research_agent.settings import Settings
from financial_research_agent.statements.contracts import (
    FinancialStatementCompany,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementResult,
    FinancialStatementSource,
    NormalizedFinancialStatement,
)

FINANCIAL_STATEMENT_STORE_VERSION = 1


class FinancialStatementStore:
    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        stale_after: timedelta = timedelta(days=30),
    ) -> None:
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self.storage_path = storage_path
        self.stale_after = stale_after
        self._results: dict[str, FinancialStatementResult] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            storage_path=settings.local_paths.data_dir / "financial_statements.json",
            stale_after=timedelta(days=settings.data_sources.financial_statement_cache_ttl_days),
        )

    def save_result(self, result: FinancialStatementResult) -> FinancialStatementResult:
        key = _result_key(result.source.provider, result.company.cik)
        with self._lock:
            self._results[key] = result
            self._save()
        return result

    def get_result(
        self,
        *,
        cik: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> FinancialStatementResult | None:
        normalized_cik = FinancialStatementCompany(cik=cik).cik
        with self._lock:
            candidates = [
                result
                for result in self._results.values()
                if result.company.cik == normalized_cik
                and (provider is None or result.source.provider == provider)
            ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda result: result.source.retrieved_at)
        return self._with_freshness(latest, now or datetime.now(UTC))

    def count(self) -> int:
        with self._lock:
            return len(self._results)

    def list(self) -> tuple[FinancialStatementResult, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._results.values(),
                    key=lambda result: result.source.retrieved_at,
                    reverse=True,
                )
            )

    def clear(self) -> int:
        with self._lock:
            deleted = len(self._results)
            self._results.clear()
            self._save()
            return deleted

    def _with_freshness(
        self,
        result: FinancialStatementResult,
        now: datetime,
    ) -> FinancialStatementResult:
        if result.source.retrieved_at + self.stale_after > now:
            return result
        warning = "Stored financial statements are stale; refresh before relying on them."
        source = replace(result.source, freshness_warning=warning)
        statements = tuple(replace(statement, source=source) for statement in result.statements)
        warnings = tuple(dict.fromkeys((*result.warnings, warning)))
        return replace(result, source=source, statements=statements, warnings=warnings)

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != FINANCIAL_STATEMENT_STORE_VERSION:
                raise ValueError("unsupported financial statement store version")
            results_payload = payload.get("results", ())
            if not isinstance(results_payload, list):
                raise ValueError("financial statement results must be a list")
            self._results = {
                _result_key(result.source.provider, result.company.cik): result
                for result in (
                    financial_statement_result_from_dict(item) for item in results_payload
                )
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"Could not load financial statement store: {self.storage_path}"
            raise ValueError(message) from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FINANCIAL_STATEMENT_STORE_VERSION,
            "results": [result.to_dict() for result in self._results.values()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def financial_statement_result_from_dict(payload: Any) -> FinancialStatementResult:
    if not isinstance(payload, dict):
        raise ValueError("financial statement result must be an object")
    company = _company_from_payload(payload["company"])
    source = _source_from_payload(payload["source"])
    statements = tuple(_statement_from_payload(item) for item in payload.get("statements", ()))
    warnings = tuple(str(item) for item in payload.get("warnings", ()))
    return FinancialStatementResult(
        company=company,
        statements=statements,
        source=source,
        warnings=warnings,
    )


def _company_from_payload(payload: Any) -> FinancialStatementCompany:
    if not isinstance(payload, dict):
        raise ValueError("financial statement company must be an object")
    return FinancialStatementCompany(
        cik=str(payload["cik"]),
        company_id=_optional_payload_text(payload, "company_id"),
        legal_name=_optional_payload_text(payload, "legal_name"),
    )


def _statement_from_payload(payload: Any) -> NormalizedFinancialStatement:
    if not isinstance(payload, dict):
        raise ValueError("financial statement must be an object")
    return NormalizedFinancialStatement(
        id=str(payload["id"]),
        company=_company_from_payload(payload["company"]),
        statement_type=FinancialStatementType(str(payload["statement_type"])),
        period=_period_from_payload(payload["period"]),
        currency=str(payload["currency"]),
        line_items={
            str(key): Decimal(str(value)) for key, value in payload.get("line_items", {}).items()
        },
        source=_source_from_payload(payload["source"]),
    )


def _period_from_payload(payload: Any) -> FinancialStatementPeriod:
    if not isinstance(payload, dict):
        raise ValueError("financial statement period must be an object")
    period_start = payload.get("period_start")
    filed_at = payload.get("filed_at")
    return FinancialStatementPeriod(
        fiscal_year=int(payload["fiscal_year"]),
        fiscal_period=str(payload["fiscal_period"]),
        period_type=FinancialStatementPeriodType(str(payload["period_type"])),
        period_start=date.fromisoformat(period_start) if isinstance(period_start, str) else None,
        period_end=date.fromisoformat(str(payload["period_end"])),
        form=_optional_payload_text(payload, "form"),
        accession_number=_optional_payload_text(payload, "accession_number"),
        filed_at=date.fromisoformat(filed_at) if isinstance(filed_at, str) else None,
    )


def _source_from_payload(payload: Any) -> FinancialStatementSource:
    if not isinstance(payload, dict):
        raise ValueError("financial statement source must be an object")
    data_as_of = payload.get("data_as_of")
    return FinancialStatementSource(
        provider=str(payload["provider"]),
        provider_status=str(payload["provider_status"]),
        source_url=str(payload["source_url"]),
        retrieved_at=_datetime_from_payload(payload["retrieved_at"]),
        data_as_of=date.fromisoformat(data_as_of) if isinstance(data_as_of, str) else None,
        attribution=str(payload["attribution"]),
        freshness_warning=_optional_payload_text(payload, "freshness_warning"),
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


def _result_key(provider: str, cik: str) -> str:
    normalized_cik = FinancialStatementCompany(cik=cik).cik
    return f"{_require_text('provider', provider)}:{normalized_cik}"


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
