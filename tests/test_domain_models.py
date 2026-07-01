from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from financial_research_agent.domain import (
    AgentOutput,
    Company,
    EvidenceItem,
    EvidenceSourceType,
    Exchange,
    Filing,
    FilingType,
    FinancialStatement,
    FinancialStatementType,
    IssueSeverity,
    PriceBar,
    ResearchIssue,
    ResearchIssueCode,
    ResearchRun,
    ResearchRunStatus,
    Security,
)


def test_company_security_and_exchange_normalize_identifiers() -> None:
    company = Company(
        id=" company:novo ",
        legal_name=" Novo Nordisk A/S ",
        aliases=["Novo", "NVO"],
        country_code="dk",
        lei=" 549300DAQ1CVT6CXN342 ",
    )
    exchange = Exchange(
        id="exchange:xnas",
        name="Nasdaq Copenhagen",
        mic="xcse",
        country_code="dk",
        currency="dkk",
    )
    security = Security(
        id="security:novo-b",
        company_id=company.id,
        exchange_id=exchange.id,
        ticker="novo-b",
        name="Novo Nordisk B",
        currency="dkk",
        isin=" dk0062498333 ",
    )

    assert company.id == "company:novo"
    assert company.aliases == ("Novo", "NVO")
    assert company.country_code == "DK"
    assert exchange.mic == "XCSE"
    assert security.ticker == "NOVO-B"
    assert security.currency == "DKK"
    assert security.isin == "DK0062498333"


def test_domain_models_are_frozen() -> None:
    company = Company(id="company:nvo", legal_name="Novo Nordisk A/S")

    with pytest.raises(FrozenInstanceError):
        company.legal_name = "Changed"  # type: ignore[misc]


def test_required_text_fields_reject_blank_values() -> None:
    with pytest.raises(ValueError, match="id is required"):
        Company(id=" ", legal_name="Novo Nordisk A/S")

    with pytest.raises(ValueError, match="ticker is required"):
        Security(
            id="security:novo-b",
            company_id="company:novo",
            exchange_id="exchange:xcse",
            ticker=" ",
            name="Novo Nordisk B",
            currency="DKK",
        )

    with pytest.raises(ValueError, match="aliases must be an iterable"):
        Company(id="company:nvo", legal_name="Novo Nordisk A/S", aliases="Novo")  # type: ignore[arg-type]


def test_financial_statement_freezes_line_items_and_validates_period() -> None:
    statement = FinancialStatement(
        id="statement:novo:2025:income",
        company_id="company:novo",
        statement_type=FinancialStatementType.INCOME_STATEMENT,
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year=2025,
        currency="dkk",
        line_items={"revenue": Decimal("290403"), "operating_income": Decimal("128339")},
        source_evidence_ids=["evidence:annual-report"],
    )

    assert dict(statement.line_items) == {
        "revenue": Decimal("290403"),
        "operating_income": Decimal("128339"),
    }
    assert statement.currency == "DKK"
    assert statement.source_evidence_ids == ("evidence:annual-report",)

    with pytest.raises(TypeError):
        statement.line_items["revenue"] = Decimal("0")  # type: ignore[index]

    with pytest.raises(ValueError, match="period_start"):
        FinancialStatement(
            id="statement:bad",
            company_id="company:novo",
            statement_type=FinancialStatementType.INCOME_STATEMENT,
            period_start=date(2025, 12, 31),
            period_end=date(2025, 1, 1),
            fiscal_year=2025,
            currency="DKK",
            line_items={"revenue": Decimal("1")},
        )

    with pytest.raises(ValueError, match="fiscal_year"):
        FinancialStatement(
            id="statement:bad-year",
            company_id="company:novo",
            statement_type=FinancialStatementType.INCOME_STATEMENT,
            period_start=None,
            period_end=date(2025, 12, 31),
            fiscal_year=0,
            currency="DKK",
            line_items={"revenue": Decimal("1")},
        )


def test_price_bar_uses_decimal_values_and_validates_ranges() -> None:
    bar = PriceBar(
        id="price:novo-b:2026-07-01",
        security_id="security:novo-b",
        priced_at=date(2026, 7, 1),
        currency="dkk",
        open="420.1",
        high=Decimal("431.2"),
        low=Decimal("418.0"),
        close=Decimal("430.0"),
        adjusted_close="430.0",
        volume=1_250_000,
        source="test-source",
    )

    assert bar.open == Decimal("420.1")
    assert bar.adjusted_close == Decimal("430.0")
    assert bar.currency == "DKK"

    with pytest.raises(ValueError, match="low"):
        PriceBar(
            id="price:bad",
            security_id="security:novo-b",
            priced_at=date(2026, 7, 1),
            currency="DKK",
            open=Decimal("1"),
            high=Decimal("1"),
            low=Decimal("2"),
            close=Decimal("1"),
            volume=0,
        )


def test_evidence_agent_output_and_research_run_keep_references() -> None:
    retrieved_at = datetime(2026, 7, 1, tzinfo=UTC)
    evidence = EvidenceItem(
        id="evidence:annual-report",
        source_type=EvidenceSourceType.FILING,
        title="Annual report",
        retrieved_at=retrieved_at,
        source_url="https://example.invalid/report.pdf",
        location="p. 10",
        metadata={"period": "FY2025"},
    )
    issue = ResearchIssue(
        code=ResearchIssueCode.PARTIAL_RESEARCH,
        message="News agent did not run.",
        severity=IssueSeverity.WARNING,
        source="orchestrator",
    )
    output = AgentOutput(
        id="agent-output:financials",
        agent_name="financial_report_analyst",
        created_at=retrieved_at,
        summary="Revenue increased.",
        findings=["Revenue grew year over year."],
        evidence_ids=[evidence.id],
        issues=[issue],
    )
    run = ResearchRun(
        id="research-run:novo",
        query="Novo Nordisk",
        created_at=retrieved_at,
        status=ResearchRunStatus.PARTIAL,
        company_id="company:novo",
        security_ids=["security:novo-b"],
        evidence_ids=[evidence.id],
        agent_output_ids=[output.id],
        issues=[issue],
        final_answer="Partial overview available.",
    )

    assert evidence.source_type == EvidenceSourceType.FILING
    assert dict(evidence.metadata) == {"period": "FY2025"}
    with pytest.raises(TypeError):
        evidence.metadata["period"] = "FY2024"  # type: ignore[index]
    assert output.evidence_ids == ("evidence:annual-report",)
    assert output.issues == (issue,)
    assert run.status == ResearchRunStatus.PARTIAL
    assert run.agent_output_ids == ("agent-output:financials",)

    with pytest.raises(ValueError, match="issues\\[0\\]"):
        AgentOutput(
            id="agent-output:bad",
            agent_name="financial_report_analyst",
            created_at=retrieved_at,
            summary="Bad issue.",
            issues=["not-an-issue"],  # type: ignore[list-item]
        )


def test_research_issue_codes_cover_milestone_error_model() -> None:
    assert {
        ResearchIssueCode.PROVIDER_UNAVAILABLE,
        ResearchIssueCode.STALE_DATA,
        ResearchIssueCode.MISSING_TICKER,
        ResearchIssueCode.AMBIGUOUS_ENTITY,
        ResearchIssueCode.PARTIAL_RESEARCH,
    } == set(ResearchIssueCode)

    issue = ResearchIssue.provider_unavailable("offline-test")

    assert issue.code == ResearchIssueCode.PROVIDER_UNAVAILABLE
    assert issue.severity == IssueSeverity.ERROR
    assert issue.source == "offline-test"


def test_filing_keeps_source_identity() -> None:
    filing = Filing(
        id="filing:novo:fy2025",
        company_id="company:novo",
        filing_type=FilingType.ANNUAL_REPORT,
        title="Annual Report 2025",
        period_end=date(2025, 12, 31),
        published_at=datetime(2026, 2, 1, tzinfo=UTC),
        source_url="https://example.invalid/annual-report",
        source_id="annual-report-2025",
    )

    assert filing.filing_type == FilingType.ANNUAL_REPORT
    assert filing.source_id == "annual-report-2025"
