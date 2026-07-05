from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any

import httpx

from financial_research_agent.domain import FinancialStatementType
from financial_research_agent.statements.contracts import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementProviderName,
    FinancialStatementResult,
    FinancialStatementSource,
    NormalizedFinancialStatement,
)

SEC_COMPANY_FACTS_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
SEC_COMPANY_FACTS_PROVIDER = FinancialStatementProviderName.SEC_COMPANY_FACTS.value
SEC_COMPANY_FACTS_STATUS = "official"
SEC_COMPANY_FACTS_ATTRIBUTION = "U.S. Securities and Exchange Commission EDGAR XBRL data"
SEC_COMPANY_FACTS_CURRENCY = "USD"
SEC_COMPANY_FACTS_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})

SEC_COMPANY_FACTS_WARNING = (
    "SEC companyfacts includes standardized non-custom taxonomy facts only; issuer-specific "
    "extension tags and full statement presentation are not ingested in this milestone."
)


@dataclass(frozen=True, slots=True)
class _ConceptDefinition:
    line_item: str
    concepts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedFact:
    line_item: str
    concept: str
    concept_priority: int
    value: Decimal
    fiscal_year: int
    fiscal_period: str
    period_start: date | None
    period_end: date
    form: str | None
    accession_number: str | None
    filed_at: date | None


STATEMENT_CONCEPTS: Mapping[FinancialStatementType, tuple[_ConceptDefinition, ...]] = {
    FinancialStatementType.INCOME_STATEMENT: (
        _ConceptDefinition(
            "revenues",
            (
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ),
        ),
        _ConceptDefinition("cost_of_revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        _ConceptDefinition("gross_profit", ("GrossProfit",)),
        _ConceptDefinition("operating_income_loss", ("OperatingIncomeLoss",)),
        _ConceptDefinition("net_income_loss", ("NetIncomeLoss", "ProfitLoss")),
    ),
    FinancialStatementType.BALANCE_SHEET: (
        _ConceptDefinition("assets", ("Assets",)),
        _ConceptDefinition("assets_current", ("AssetsCurrent",)),
        _ConceptDefinition(
            "cash_and_cash_equivalents",
            ("CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks"),
        ),
        _ConceptDefinition("liabilities", ("Liabilities",)),
        _ConceptDefinition("liabilities_current", ("LiabilitiesCurrent",)),
        _ConceptDefinition(
            "stockholders_equity",
            (
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ),
        ),
    ),
    FinancialStatementType.CASH_FLOW: (
        _ConceptDefinition(
            "net_cash_provided_by_operating_activities",
            ("NetCashProvidedByUsedInOperatingActivities",),
        ),
        _ConceptDefinition(
            "net_cash_provided_by_investing_activities",
            ("NetCashProvidedByUsedInInvestingActivities",),
        ),
        _ConceptDefinition(
            "net_cash_provided_by_financing_activities",
            ("NetCashProvidedByUsedInFinancingActivities",),
        ),
        _ConceptDefinition(
            "capital_expenditures",
            ("PaymentsToAcquirePropertyPlantAndEquipment",),
        ),
    ),
}


class SECCompanyFactsProvider:
    def __init__(
        self,
        *,
        base_url: str = SEC_COMPANY_FACTS_BASE_URL,
        user_agent: str = (
            "financial-research-agent/0.1 local-research contact@financial-research-agent.local"
        ),
        http_client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.base_url = _require_text("base_url", base_url).rstrip("/")
        self.user_agent = _require_text("user_agent", user_agent)
        self._http_client = http_client
        self._now = now or (lambda: datetime.now(UTC))

    async def fetch_statements(
        self,
        company: FinancialStatementCompany,
        *,
        fiscal_years: int = 3,
    ) -> FinancialStatementResult:
        if fiscal_years <= 0 or fiscal_years > 10:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.INVALID_REQUEST,
                message="fiscal_years must be between 1 and 10",
                provider=SEC_COMPANY_FACTS_PROVIDER,
            )
        payload = await self._get_companyfacts(company)
        return _normalize_companyfacts(
            payload,
            requested_company=company,
            fiscal_years=fiscal_years,
            source_url=self._source_url(company),
            retrieved_at=self._now(),
        )

    async def _get_companyfacts(self, company: FinancialStatementCompany) -> Mapping[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        try:
            if self._http_client is None:
                async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                    response = await client.get(self._source_url(company))
            else:
                response = await self._http_client.get(self._source_url(company), headers=headers)
        except httpx.TimeoutException as exc:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.TIMEOUT,
                message="Timed out while fetching SEC companyfacts.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.PROVIDER_UNAVAILABLE,
                message=f"SEC companyfacts source is unavailable: {exc}",
                provider=SEC_COMPANY_FACTS_PROVIDER,
                retryable=True,
            ) from exc
        if response.status_code == 404:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.NOT_FOUND,
                message=f"No SEC companyfacts found for CIK {company.padded_cik}.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
            )
        if response.status_code == 429:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.RATE_LIMITED,
                message="SEC companyfacts source rate limited the request.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 500:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.PROVIDER_UNAVAILABLE,
                message=f"SEC companyfacts source returned HTTP {response.status_code}.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
                retryable=True,
            )
        if response.status_code >= 400:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.INVALID_REQUEST,
                message=f"SEC companyfacts source returned HTTP {response.status_code}.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.MALFORMED_RESPONSE,
                message="SEC companyfacts source returned malformed JSON.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
            ) from exc
        if not isinstance(payload, Mapping):
            raise FinancialStatementError(
                code=FinancialStatementErrorCode.MALFORMED_RESPONSE,
                message="SEC companyfacts payload must be a JSON object.",
                provider=SEC_COMPANY_FACTS_PROVIDER,
            )
        return payload

    def _source_url(self, company: FinancialStatementCompany) -> str:
        return f"{self.base_url}/CIK{company.padded_cik}.json"


def _normalize_companyfacts(
    payload: Mapping[str, Any],
    *,
    requested_company: FinancialStatementCompany,
    fiscal_years: int,
    source_url: str,
    retrieved_at: datetime,
) -> FinancialStatementResult:
    company = _company_from_payload(payload, requested_company)
    selected_facts: dict[tuple[str, int], _NormalizedFact] = {}
    duplicate_count = 0
    ignored_units: set[str] = set()
    for definitions in STATEMENT_CONCEPTS.values():
        for definition in definitions:
            facts, units = _facts_for_definition(payload, definition)
            ignored_units.update(units - {SEC_COMPANY_FACTS_CURRENCY})
            for fact in facts:
                key = (definition.line_item, fact.fiscal_year)
                existing = selected_facts.get(key)
                if existing is not None:
                    duplicate_count += 1
                if existing is None or _is_better_fact(fact, existing):
                    selected_facts[key] = fact

    fiscal_year_order = _latest_fiscal_years(selected_facts.values(), fiscal_years)
    if not fiscal_year_order:
        raise FinancialStatementError(
            code=FinancialStatementErrorCode.NOT_FOUND,
            message=(
                "SEC companyfacts did not include supported annual USD statement facts "
                f"for CIK {company.padded_cik}."
            ),
            provider=SEC_COMPANY_FACTS_PROVIDER,
        )

    source = _source_metadata(
        source_url=source_url,
        retrieved_at=retrieved_at,
        data_as_of=_latest_filed_at(selected_facts.values()),
    )
    statements: list[NormalizedFinancialStatement] = []
    for fiscal_year in fiscal_year_order:
        year_facts = {
            line_item: fact
            for (line_item, fact_year), fact in selected_facts.items()
            if fact_year == fiscal_year
        }
        statements.extend(_statements_for_year(company, fiscal_year, year_facts, source))

    warnings = [SEC_COMPANY_FACTS_WARNING]
    if len(fiscal_year_order) < fiscal_years:
        warnings.append(
            f"Only {len(fiscal_year_order)} annual fiscal period(s) could be normalized."
        )
    missing_types = _missing_statement_types(statements)
    if missing_types:
        warnings.append(f"Missing normalized statement type(s): {', '.join(missing_types)}.")
    if duplicate_count:
        warnings.append(
            "Duplicate or restated SEC facts were resolved by selecting the latest annual fact."
        )
    if ignored_units:
        warnings.append(
            "Non-USD or non-statement SEC fact units were ignored: "
            f"{', '.join(sorted(ignored_units))}."
        )

    return FinancialStatementResult(
        company=company,
        statements=tuple(statements),
        source=source,
        warnings=tuple(warnings),
    )


def _company_from_payload(
    payload: Mapping[str, Any],
    requested_company: FinancialStatementCompany,
) -> FinancialStatementCompany:
    cik = str(payload.get("cik", requested_company.cik))
    name = payload.get("entityName")
    legal_name = (
        str(name) if isinstance(name, str) and name.strip() else requested_company.legal_name
    )
    return FinancialStatementCompany(
        cik=cik,
        company_id=requested_company.company_id,
        legal_name=legal_name,
    )


def _facts_for_definition(
    payload: Mapping[str, Any],
    definition: _ConceptDefinition,
) -> tuple[tuple[_NormalizedFact, ...], set[str]]:
    facts: list[_NormalizedFact] = []
    seen_units: set[str] = set()
    for priority, concept in enumerate(definition.concepts):
        concept_payload = _concept_payload(payload, concept)
        if concept_payload is None:
            continue
        units = concept_payload.get("units")
        if not isinstance(units, Mapping):
            continue
        seen_units.update(str(unit) for unit in units)
        usd_facts = units.get(SEC_COMPANY_FACTS_CURRENCY)
        if not isinstance(usd_facts, Iterable) or isinstance(usd_facts, (str, bytes)):
            continue
        for fact_payload in usd_facts:
            fact = _fact_from_payload(
                fact_payload,
                definition=definition,
                concept=concept,
                concept_priority=priority,
            )
            if fact is not None:
                facts.append(fact)
    return tuple(facts), seen_units


def _concept_payload(payload: Mapping[str, Any], concept: str) -> Mapping[str, Any] | None:
    facts = payload.get("facts")
    if not isinstance(facts, Mapping):
        raise FinancialStatementError(
            code=FinancialStatementErrorCode.MALFORMED_RESPONSE,
            message="SEC companyfacts payload does not include a facts object.",
            provider=SEC_COMPANY_FACTS_PROVIDER,
        )
    us_gaap = facts.get("us-gaap")
    if not isinstance(us_gaap, Mapping):
        return None
    concept_payload = us_gaap.get(concept)
    return concept_payload if isinstance(concept_payload, Mapping) else None


def _fact_from_payload(
    payload: Any,
    *,
    definition: _ConceptDefinition,
    concept: str,
    concept_priority: int,
) -> _NormalizedFact | None:
    if not isinstance(payload, Mapping):
        return None
    form = _optional_text(payload.get("form"))
    if form not in SEC_COMPANY_FACTS_FORMS:
        return None
    fiscal_period = _optional_text(payload.get("fp"))
    if fiscal_period != "FY":
        return None
    try:
        fiscal_year = int(payload["fy"])
        period_end = date.fromisoformat(str(payload["end"]))
        value = Decimal(str(payload["val"]))
    except KeyError, TypeError, ValueError, InvalidOperation:
        return None
    period_start_value = payload.get("start")
    filed_value = payload.get("filed")
    period_start = _date_or_none(period_start_value)
    filed_at = _date_or_none(filed_value)
    return _NormalizedFact(
        line_item=definition.line_item,
        concept=concept,
        concept_priority=concept_priority,
        value=value,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_start=period_start,
        period_end=period_end,
        form=form,
        accession_number=_optional_text(payload.get("accn")),
        filed_at=filed_at,
    )


def _statements_for_year(
    company: FinancialStatementCompany,
    fiscal_year: int,
    year_facts: Mapping[str, _NormalizedFact],
    source: FinancialStatementSource,
) -> tuple[NormalizedFinancialStatement, ...]:
    statements: list[NormalizedFinancialStatement] = []
    for statement_type, definitions in STATEMENT_CONCEPTS.items():
        line_items = {
            definition.line_item: year_facts[definition.line_item].value
            for definition in definitions
            if definition.line_item in year_facts
        }
        if not line_items:
            continue
        period = _period_for_facts(tuple(year_facts[item] for item in line_items))
        statements.append(
            _statement(
                company=company,
                statement_type=statement_type,
                period=period,
                line_items=line_items,
                source=source,
            )
        )
    ratios = _ratios_for_year(year_facts)
    if ratios:
        ratio_facts = tuple(year_facts.values())
        statements.append(
            _statement(
                company=company,
                statement_type=FinancialStatementType.KEY_RATIOS,
                period=_period_for_facts(ratio_facts),
                line_items=ratios,
                source=source,
            )
        )
    return tuple(statements)


def _ratios_for_year(facts: Mapping[str, _NormalizedFact]) -> dict[str, Decimal]:
    ratios: dict[str, Decimal] = {}
    revenues = _value(facts, "revenues")
    if revenues is not None and revenues != 0:
        _ratio(ratios, "gross_margin", _value(facts, "gross_profit"), revenues)
        _ratio(ratios, "operating_margin", _value(facts, "operating_income_loss"), revenues)
        _ratio(ratios, "net_margin", _value(facts, "net_income_loss"), revenues)
    current_liabilities = _value(facts, "liabilities_current")
    current_assets = _value(facts, "assets_current")
    if current_assets is not None and current_liabilities not in (None, Decimal("0")):
        _ratio(ratios, "current_ratio", current_assets, current_liabilities)
    operating_cash_flow = _value(facts, "net_cash_provided_by_operating_activities")
    capital_expenditures = _value(facts, "capital_expenditures")
    if operating_cash_flow is not None and capital_expenditures is not None:
        ratios["free_cash_flow_proxy"] = operating_cash_flow - capital_expenditures
    return ratios


def _ratio(
    ratios: dict[str, Decimal],
    name: str,
    numerator: Decimal | None,
    denominator: Decimal,
) -> None:
    if numerator is None:
        return
    try:
        ratios[name] = numerator / denominator
    except DivisionByZero, InvalidOperation:
        return


def _value(facts: Mapping[str, _NormalizedFact], line_item: str) -> Decimal | None:
    fact = facts.get(line_item)
    return fact.value if fact is not None else None


def _statement(
    *,
    company: FinancialStatementCompany,
    statement_type: FinancialStatementType,
    period: FinancialStatementPeriod,
    line_items: Mapping[str, Decimal],
    source: FinancialStatementSource,
) -> NormalizedFinancialStatement:
    return NormalizedFinancialStatement(
        id=f"sec:companyfacts:{company.padded_cik}:{statement_type.value}:{period.fiscal_year}",
        company=company,
        statement_type=statement_type,
        period=period,
        currency=SEC_COMPANY_FACTS_CURRENCY,
        line_items=line_items,
        source=source,
    )


def _period_for_facts(facts: tuple[_NormalizedFact, ...]) -> FinancialStatementPeriod:
    ordered = sorted(facts, key=lambda fact: (fact.period_end, fact.filed_at or date.min))
    latest = ordered[-1]
    starts = [fact.period_start for fact in ordered if fact.period_start is not None]
    return FinancialStatementPeriod(
        fiscal_year=latest.fiscal_year,
        fiscal_period=latest.fiscal_period,
        period_type=FinancialStatementPeriodType.ANNUAL,
        period_start=min(starts) if starts else None,
        period_end=latest.period_end,
        form=latest.form,
        accession_number=latest.accession_number,
        filed_at=latest.filed_at,
    )


def _latest_fiscal_years(facts: Iterable[_NormalizedFact], limit: int) -> tuple[int, ...]:
    years: dict[int, date] = {}
    for fact in facts:
        current = years.get(fact.fiscal_year)
        if current is None or fact.period_end > current:
            years[fact.fiscal_year] = fact.period_end
    sorted_years = sorted(years.items(), key=lambda item: item[1], reverse=True)
    return tuple(year for year, _end in sorted_years[:limit])


def _latest_filed_at(facts: Iterable[_NormalizedFact]) -> date | None:
    filed_dates = [fact.filed_at for fact in facts if fact.filed_at is not None]
    return max(filed_dates) if filed_dates else None


def _missing_statement_types(statements: Iterable[NormalizedFinancialStatement]) -> tuple[str, ...]:
    present = {statement.statement_type for statement in statements}
    required = {
        FinancialStatementType.INCOME_STATEMENT,
        FinancialStatementType.BALANCE_SHEET,
        FinancialStatementType.CASH_FLOW,
        FinancialStatementType.KEY_RATIOS,
    }
    return tuple(statement_type.value for statement_type in sorted(required - present, key=str))


def _is_better_fact(candidate: _NormalizedFact, existing: _NormalizedFact) -> bool:
    if candidate.period_end != existing.period_end:
        return candidate.period_end > existing.period_end
    if candidate.concept_priority != existing.concept_priority:
        return candidate.concept_priority < existing.concept_priority
    return (candidate.filed_at or date.min) > (existing.filed_at or date.min)


def _source_metadata(
    *,
    source_url: str,
    retrieved_at: datetime,
    data_as_of: date | None,
) -> FinancialStatementSource:
    return FinancialStatementSource(
        provider=SEC_COMPANY_FACTS_PROVIDER,
        provider_status=SEC_COMPANY_FACTS_STATUS,
        source_url=source_url,
        retrieved_at=retrieved_at,
        data_as_of=data_as_of,
        attribution=SEC_COMPANY_FACTS_ATTRIBUTION,
    )


def _date_or_none(value: object) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
