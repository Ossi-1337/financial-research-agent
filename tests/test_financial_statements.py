from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementStore,
    FinancialStatementType,
    SECCompanyFactsProvider,
)


def _fact(fiscal_year: int, end: str, filed: str, value: int) -> dict[str, object]:
    return {
        "fy": fiscal_year,
        "fp": "FY",
        "form": "10-K",
        "start": f"{fiscal_year - 1}-09-28",
        "end": end,
        "filed": filed,
        "val": value,
        "accn": f"{fiscal_year}-fixture",
    }


def _instant_fact(fiscal_year: int, end: str, filed: str, value: int) -> dict[str, object]:
    return {
        "fy": fiscal_year,
        "fp": "FY",
        "form": "10-K",
        "end": end,
        "filed": filed,
        "val": value,
        "accn": f"{fiscal_year}-fixture",
    }


SEC_FACTS_FIXTURE = {
    "cik": 320193,
    "entityName": "TEST TOOL OUTPUT APPLE INC.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        _fact(2025, "2025-09-27", "2025-10-24", 900),
                        _fact(2025, "2025-09-27", "2025-10-31", 1000),
                    ],
                    "USD/shares": [],
                }
            },
            "GrossProfit": {"units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", 600)]}},
            "OperatingIncomeLoss": {
                "units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", 400)]}
            },
            "NetIncomeLoss": {"units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", 250)]}},
            "Assets": {"units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 2000)]}},
            "AssetsCurrent": {
                "units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 800)]}
            },
            "CashAndCashEquivalentsAtCarryingValue": {
                "units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 200)]}
            },
            "Liabilities": {
                "units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 1000)]}
            },
            "LiabilitiesCurrent": {
                "units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 400)]}
            },
            "StockholdersEquity": {
                "units": {"USD": [_instant_fact(2025, "2025-09-27", "2025-10-31", 1000)]}
            },
            "NetCashProvidedByUsedInOperatingActivities": {
                "units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", 300)]}
            },
            "NetCashProvidedByUsedInInvestingActivities": {
                "units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", -100)]}
            },
            "NetCashProvidedByUsedInFinancingActivities": {
                "units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", -50)]}
            },
            "PaymentsToAcquirePropertyPlantAndEquipment": {
                "units": {"USD": [_fact(2025, "2025-09-27", "2025-10-31", 80)]}
            },
        }
    },
}


def _ifrs_fact(value: int, *, start: str | None = "2025-01-01") -> dict[str, object]:
    payload: dict[str, object] = {
        "fy": 2025,
        "fp": "FY",
        "form": "20-F",
        "end": "2025-12-31",
        "filed": "2026-02-04",
        "val": value,
        "accn": "0000353278-26-000001",
    }
    if start is not None:
        payload["start"] = start
    return payload


IFRS_FACTS_FIXTURE = {
    "cik": 353278,
    "entityName": "TEST TOOL OUTPUT NOVO NORDISK A S",
    "facts": {
        "ifrs-full": {
            "Revenue": {
                "units": {
                    "DKK": [_ifrs_fact(300_000)],
                    "USD": [_ifrs_fact(40_000)],
                    "shares": [],
                }
            },
            "GrossProfit": {"units": {"DKK": [_ifrs_fact(250_000)]}},
            "ProfitLossFromOperatingActivities": {"units": {"DKK": [_ifrs_fact(120_000)]}},
            "ProfitLoss": {"units": {"DKK": [_ifrs_fact(90_000)]}},
            "Assets": {"units": {"DKK": [_ifrs_fact(500_000, start=None)]}},
            "CurrentAssets": {"units": {"DKK": [_ifrs_fact(200_000, start=None)]}},
            "CashAndCashEquivalents": {"units": {"DKK": [_ifrs_fact(80_000, start=None)]}},
            "Liabilities": {"units": {"DKK": [_ifrs_fact(250_000, start=None)]}},
            "CurrentLiabilities": {"units": {"DKK": [_ifrs_fact(100_000, start=None)]}},
            "Equity": {"units": {"DKK": [_ifrs_fact(250_000, start=None)]}},
            "CashFlowsFromUsedInOperatingActivities": {"units": {"DKK": [_ifrs_fact(110_000)]}},
            "CashFlowsFromUsedInInvestingActivities": {"units": {"DKK": [_ifrs_fact(-60_000)]}},
            "CashFlowsFromUsedInFinancingActivities": {"units": {"DKK": [_ifrs_fact(-30_000)]}},
            "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": {
                "units": {"DKK": [_ifrs_fact(45_000)]}
            },
        }
    },
}


def test_financial_statement_company_contract_is_immutable_and_normalizes_cik() -> None:
    company = FinancialStatementCompany(cik="CIK0000320193", legal_name="Test Company")

    assert company.cik == "320193"
    assert company.padded_cik == "0000320193"
    with pytest.raises(FrozenInstanceError):
        company.cik = "1"  # type: ignore[misc]


def test_sec_companyfacts_provider_normalizes_annual_statements_and_ratios() -> None:
    provider = SECCompanyFactsProvider(
        base_url="https://data.sec.test/api/xbrl/companyfacts",
        http_client=_client_with_json(SEC_FACTS_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    result = asyncio.run(
        provider.fetch_statements(
            FinancialStatementCompany(cik="0000320193", company_id="sec:cik:0000320193"),
            fiscal_years=1,
        )
    )

    assert result.company.legal_name == "TEST TOOL OUTPUT APPLE INC."
    assert result.source.provider == "sec-companyfacts"
    assert result.source.source_url.endswith("/CIK0000320193.json")
    assert result.source.data_as_of == date(2025, 10, 31)
    assert {statement.statement_type for statement in result.statements} == {
        FinancialStatementType.INCOME_STATEMENT,
        FinancialStatementType.BALANCE_SHEET,
        FinancialStatementType.CASH_FLOW,
        FinancialStatementType.KEY_RATIOS,
    }
    income = _statement(result, FinancialStatementType.INCOME_STATEMENT)
    ratios = _statement(result, FinancialStatementType.KEY_RATIOS)
    assert income.line_items["revenues"] == Decimal("1000")
    assert income.period.fiscal_year == 2025
    assert ratios.line_items["gross_margin"] == Decimal("0.6")
    assert ratios.line_items["current_ratio"] == Decimal("2")
    assert ratios.line_items["free_cash_flow_proxy"] == Decimal("220")
    assert "Duplicate or restated SEC facts" in " ".join(result.warnings)
    assert "USD/shares" in " ".join(result.warnings)


def test_sec_companyfacts_provider_warns_when_requested_periods_are_missing() -> None:
    provider = SECCompanyFactsProvider(
        http_client=_client_with_json(SEC_FACTS_FIXTURE),
        now=lambda: datetime(2026, 7, 4, tzinfo=UTC),
    )

    result = asyncio.run(
        provider.fetch_statements(FinancialStatementCompany(cik="320193"), fiscal_years=3)
    )

    assert "Only 1 annual fiscal period(s)" in " ".join(result.warnings)


def test_sec_companyfacts_provider_normalizes_ifrs_dkk_with_provenance() -> None:
    provider = SECCompanyFactsProvider(
        http_client=_client_with_json(IFRS_FACTS_FIXTURE),
        now=lambda: datetime(2026, 2, 5, tzinfo=UTC),
    )

    result = asyncio.run(
        provider.fetch_statements(FinancialStatementCompany(cik="0000353278"), fiscal_years=1)
    )
    income = _statement(result, FinancialStatementType.INCOME_STATEMENT)
    ratios = _statement(result, FinancialStatementType.KEY_RATIOS)

    assert income.currency == "DKK"
    assert income.period.form == "20-F"
    assert income.line_items["revenues"] == Decimal("300000")
    assert ratios.line_items["operating_margin"] == Decimal("0.4")
    assert result.source.taxonomy_namespaces == ("ifrs-full",)
    assert result.source.concept_mappings["revenues"] == "ifrs-full:Revenue"
    assert "Alternate currency facts were ignored" in " ".join(result.warnings)
    assert "shares" in " ".join(result.warnings)


def test_sec_companyfacts_provider_returns_not_found_for_missing_supported_facts() -> None:
    provider = SECCompanyFactsProvider(
        http_client=_client_with_json({"cik": 1, "facts": {"us-gaap": {}}}),
    )

    with pytest.raises(FinancialStatementError) as exc_info:
        asyncio.run(provider.fetch_statements(FinancialStatementCompany(cik="1")))

    assert exc_info.value.code == FinancialStatementErrorCode.NOT_FOUND


def test_sec_companyfacts_provider_maps_rate_limits_and_malformed_payloads() -> None:
    rate_limited = SECCompanyFactsProvider(
        http_client=_client_with_response(httpx.Response(429, json={"error": "slow down"})),
    )
    malformed = SECCompanyFactsProvider(
        http_client=_client_with_response(httpx.Response(200, text="not json")),
    )

    with pytest.raises(FinancialStatementError) as rate_error:
        asyncio.run(rate_limited.fetch_statements(FinancialStatementCompany(cik="1")))
    with pytest.raises(FinancialStatementError) as malformed_error:
        asyncio.run(malformed.fetch_statements(FinancialStatementCompany(cik="1")))

    assert rate_error.value.code == FinancialStatementErrorCode.RATE_LIMITED
    assert rate_error.value.retryable is True
    assert malformed_error.value.code == FinancialStatementErrorCode.MALFORMED_RESPONSE


def test_financial_statement_store_persists_and_marks_stale_results(tmp_path) -> None:
    provider = SECCompanyFactsProvider(
        http_client=_client_with_json(SEC_FACTS_FIXTURE),
        now=lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    result = asyncio.run(provider.fetch_statements(FinancialStatementCompany(cik="320193")))
    store = FinancialStatementStore(
        storage_path=tmp_path / "financial_statements.json",
        stale_after=timedelta(days=1),
    )

    store.save_result(result)
    reloaded = FinancialStatementStore(
        storage_path=tmp_path / "financial_statements.json",
        stale_after=timedelta(days=1),
    )
    fresh = reloaded.get_result(cik="0000320193", now=datetime(2026, 7, 1, tzinfo=UTC))
    stale = reloaded.get_result(cik="320193", now=datetime(2026, 7, 4, tzinfo=UTC))

    assert fresh is not None
    fresh_income = _statement(fresh, FinancialStatementType.INCOME_STATEMENT)
    assert fresh_income.line_items["revenues"] == Decimal("1000")
    assert stale is not None
    assert "Stored financial statements are stale" in stale.warnings[-1]
    assert (
        stale.source.freshness_warning
        == "Stored financial statements are stale; refresh before relying on them."
    )


def _statement(result, statement_type: FinancialStatementType):
    return next(
        statement for statement in result.statements if statement.statement_type == statement_type
    )


def _client_with_json(payload: object) -> httpx.AsyncClient:
    return _client_with_response(httpx.Response(200, json=payload))


def _client_with_response(response: httpx.Response) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: response))
