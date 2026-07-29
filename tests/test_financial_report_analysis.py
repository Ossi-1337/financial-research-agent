from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from financial_research_agent.documents import DocumentExtractionMethod, DocumentRegion
from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingIngestionResult,
    FilingSource,
    FilingStore,
)
from financial_research_agent.report_analysis import (
    ConfidenceLabel,
    FinancialReportAnalysisAgent,
    FinancialReportAnalysisCompany,
    FinancialReportAnalysisStatus,
    FinancialReportFinding,
    FinancialReportSection,
)
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementResult,
    FinancialStatementSource,
    FinancialStatementStore,
    FinancialStatementType,
    NormalizedFinancialStatement,
)

NOW = datetime(2026, 7, 5, tzinfo=UTC)


def test_financial_report_finding_contract_requires_evidence_or_limitation() -> None:
    finding = FinancialReportFinding(
        id="finding:test",
        section=FinancialReportSection.REVENUE,
        title="Revenue",
        summary="Stored test evidence was unavailable.",
        confidence=ConfidenceLabel.UNKNOWN,
        limitations=("Stored test evidence was unavailable.",),
    )

    assert finding.to_dict()["section"] == "revenue"
    with pytest.raises(FrozenInstanceError):
        finding.title = "Changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="evidence_ids or limitations"):
        FinancialReportFinding(
            id="finding:invalid",
            section=FinancialReportSection.REVENUE,
            title="Invalid",
            summary="Invalid",
            confidence=ConfidenceLabel.HIGH,
        )


def test_financial_report_analysis_agent_produces_grounded_sections() -> None:
    statement_store = FinancialStatementStore()
    filing_store = FilingStore()
    statement_store.save_result(_statement_result())
    filing_store.save_result(_filing_result())
    agent = FinancialReportAnalysisAgent(
        statement_store=statement_store,
        filing_store=filing_store,
        statement_provider="sec-companyfacts",
        filing_provider="sec-edgar",
        now=lambda: NOW,
    )

    result = agent.analyze(
        FinancialReportAnalysisCompany(
            cik="0000320193",
            company_id="fixture:company:apple",
            legal_name="TEST TOOL OUTPUT APPLE INC.",
        )
    )

    assert result.status == FinancialReportAnalysisStatus.PARTIAL
    assert {finding.section for finding in result.findings} == set(FinancialReportSection)
    assert {question.section for question in result.questions} == set(FinancialReportSection)
    assert result.source_summary["statement_count"] == "5"
    assert result.source_summary["filing_chunk_count"] == "3"
    assert len(result.citations) == len(result.evidence)
    assert all(finding.evidence_ids or finding.limitations for finding in result.findings)
    assert all(citation.source_url for citation in result.citations)
    filing_citations = [
        citation for citation in result.citations if citation.metadata.get("kind") == "filing_chunk"
    ]
    assert filing_citations
    assert all("#page=" in citation.source_url for citation in filing_citations)
    assert all("(p. " in (citation.section or "") for citation in filing_citations)

    revenue = _finding(result.findings, FinancialReportSection.REVENUE)
    assert revenue.evidence_ids
    assert revenue.prior_period_comparison is not None
    assert "increased" in revenue.prior_period_comparison
    assert "TEST TOOL OUTPUT" not in revenue.summary

    risks = _finding(result.findings, FinancialReportSection.RISKS)
    assert risks.evidence_ids
    assert risks.citation_ids
    assert "lexical ranking" in " ".join(risks.limitations)


def test_financial_report_analysis_agent_reports_no_data_without_inventing_findings() -> None:
    agent = FinancialReportAnalysisAgent(
        statement_store=FinancialStatementStore(),
        filing_store=FilingStore(),
        now=lambda: NOW,
    )

    result = agent.analyze(FinancialReportAnalysisCompany(cik="320193"))

    assert result.status == FinancialReportAnalysisStatus.NO_DATA
    assert result.citations == ()
    assert result.evidence == ()
    assert result.limitations
    assert all(finding.confidence == ConfidenceLabel.UNKNOWN for finding in result.findings)
    assert all(finding.limitations for finding in result.findings)


def test_financial_report_analysis_agent_degrades_on_store_failures() -> None:
    agent = FinancialReportAnalysisAgent(
        statement_store=FailingStore("statement storage offline"),
        filing_store=FailingStore("filing storage offline"),
        now=lambda: NOW,
    )

    result = agent.analyze(FinancialReportAnalysisCompany(cik="320193"))

    assert result.status == FinancialReportAnalysisStatus.NO_DATA
    assert "statement storage offline" in " ".join(result.limitations)
    assert "filing storage offline" in " ".join(result.limitations)


def _statement_result() -> FinancialStatementResult:
    company = FinancialStatementCompany(
        cik="320193",
        company_id="fixture:company:apple",
        legal_name="TEST TOOL OUTPUT APPLE INC.",
    )
    source = FinancialStatementSource(
        provider="sec-companyfacts",
        provider_status="test fixture",
        source_url="https://example.invalid/sec-companyfacts/CIK0000320193.json",
        retrieved_at=NOW,
        data_as_of=date(2026, 1, 31),
        attribution="test fixture",
    )
    return FinancialStatementResult(
        company=company,
        statements=(
            _statement(
                company,
                source,
                "income-2025",
                FinancialStatementType.INCOME_STATEMENT,
                2025,
                {
                    "revenues": Decimal("1000"),
                    "gross_profit": Decimal("600"),
                    "operating_income_loss": Decimal("400"),
                    "net_income_loss": Decimal("250"),
                },
            ),
            _statement(
                company,
                source,
                "income-2024",
                FinancialStatementType.INCOME_STATEMENT,
                2024,
                {"revenues": Decimal("800")},
            ),
            _statement(
                company,
                source,
                "balance-2025",
                FinancialStatementType.BALANCE_SHEET,
                2025,
                {
                    "assets_current": Decimal("800"),
                    "liabilities_current": Decimal("400"),
                    "cash_and_cash_equivalents": Decimal("200"),
                    "liabilities": Decimal("1000"),
                    "stockholders_equity": Decimal("1000"),
                },
            ),
            _statement(
                company,
                source,
                "cash-2025",
                FinancialStatementType.CASH_FLOW,
                2025,
                {"net_cash_provided_by_operating_activities": Decimal("300")},
            ),
            _statement(
                company,
                source,
                "ratios-2025",
                FinancialStatementType.KEY_RATIOS,
                2025,
                {
                    "gross_margin": Decimal("0.6"),
                    "operating_margin": Decimal("0.4"),
                    "net_margin": Decimal("0.25"),
                    "current_ratio": Decimal("2"),
                    "free_cash_flow_proxy": Decimal("220"),
                },
            ),
        ),
        source=source,
        warnings=("TEST TOOL OUTPUT statement fixture.",),
    )


def _statement(
    company: FinancialStatementCompany,
    source: FinancialStatementSource,
    statement_id: str,
    statement_type: FinancialStatementType,
    fiscal_year: int,
    line_items: dict[str, Decimal],
) -> NormalizedFinancialStatement:
    period = FinancialStatementPeriod(
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        period_type=FinancialStatementPeriodType.ANNUAL,
        period_start=date(fiscal_year - 1, 1, 1),
        period_end=date(fiscal_year, 12, 31),
        form="10-K",
        accession_number=f"fixture-{fiscal_year}",
        filed_at=date(fiscal_year + 1, 1, 31),
    )
    return NormalizedFinancialStatement(
        id=f"fixture:statement:{statement_id}",
        company=company,
        statement_type=statement_type,
        period=period,
        currency="USD",
        line_items=line_items,
        source=source,
    )


def _filing_result() -> FilingIngestionResult:
    company = FilingCompany(
        cik="320193",
        company_id="fixture:company:apple",
        legal_name="TEST TOOL OUTPUT APPLE INC.",
    )
    source = FilingSource(
        provider="sec-edgar",
        provider_status="test fixture",
        source_url="https://example.invalid/submissions/CIK0000320193.json",
        retrieved_at=NOW,
        data_as_of=date(2026, 1, 31),
        attribution="test fixture",
    )
    filing = FilingDocument(
        id="fixture:filing:10-k",
        company=company,
        form_type="10-K",
        accession_number="0000320193-26-000001",
        filing_date=date(2026, 1, 31),
        report_date=date(2025, 12, 31),
        publication_date=date(2026, 1, 31),
        document_url="https://example.invalid/aapl-20251231.htm",
        source_url=source.source_url,
        document_format=FilingDocumentFormat.HTML,
        retrieved_at=NOW,
        local_raw_path="fixture/raw/aapl-20251231.htm",
        local_text_path="fixture/text/aapl-20251231.txt",
        source=source,
        chunk_ids=("fixture:chunk:guidance", "fixture:chunk:risks", "fixture:chunk:accounting"),
    )
    chunks = (
        _chunk(
            filing,
            "guidance",
            0,
            "TEST TOOL OUTPUT guidance outlook expects product demand to vary by region.",
            "Item 7. Management Discussion",
        ),
        _chunk(
            filing,
            "risks",
            1,
            "TEST TOOL OUTPUT risk factors include competition and uncertain supply conditions.",
            "Item 1A. Risk Factors",
        ),
        _chunk(
            filing,
            "accounting",
            2,
            "TEST TOOL OUTPUT critical accounting estimate language discusses impairment and tax.",
            "Critical Accounting Estimates",
        ),
    )
    return FilingIngestionResult(
        company=company,
        filings=(filing,),
        chunks=chunks,
        source=source,
        warnings=("TEST TOOL OUTPUT filing fixture.",),
    )


def _chunk(
    filing: FilingDocument,
    label: str,
    index: int,
    text: str,
    heading: str,
) -> FilingChunk:
    return FilingChunk(
        id=f"fixture:chunk:{label}",
        filing_id=filing.id,
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
        source_url=filing.document_url,
        accession_number=filing.accession_number,
        form_type=filing.form_type,
        section_heading=heading,
        metadata={"fixture": "true"},
        source_region=DocumentRegion(
            page_number=index + 1,
            left=0.1,
            top=0.1,
            right=0.9,
            bottom=0.9,
        ),
        extraction_method=DocumentExtractionMethod.PDF_NATIVE_TEXT,
    )


def _finding(
    findings: tuple[FinancialReportFinding, ...],
    section: FinancialReportSection,
) -> FinancialReportFinding:
    return next(finding for finding in findings if finding.section == section)


class FailingStore:
    def __init__(self, message: str) -> None:
        self._message = message

    def get_result(self, **_kwargs):
        raise ValueError(self._message)
