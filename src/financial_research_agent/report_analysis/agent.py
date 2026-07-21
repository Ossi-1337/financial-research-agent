from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol
from uuid import uuid4

from financial_research_agent.domain import FinancialStatementType
from financial_research_agent.filings import FilingChunk, FilingIngestionResult
from financial_research_agent.report_analysis.contracts import (
    NO_RECOMMENDATION_NOTICE,
    ConfidenceLabel,
    FinancialReportAnalysisCompany,
    FinancialReportAnalysisResult,
    FinancialReportAnalysisStatus,
    FinancialReportFinding,
    FinancialReportQuestion,
    FinancialReportSection,
)
from financial_research_agent.reports import Citation, EvidenceSnippet
from financial_research_agent.statements import (
    FinancialStatementResult,
    NormalizedFinancialStatement,
)

REVENUE_KEYS = ("revenues", "sales_revenue_net", "revenue")
GROSS_PROFIT_KEYS = ("gross_profit",)
OPERATING_INCOME_KEYS = ("operating_income_loss", "operating_income")
NET_INCOME_KEYS = ("net_income_loss", "net_income")
OPERATING_CASH_FLOW_KEYS = ("net_cash_provided_by_operating_activities", "operating_cash_flow")
FREE_CASH_FLOW_KEYS = ("free_cash_flow_proxy", "free_cash_flow")
CASH_KEYS = ("cash_and_cash_equivalents", "cash_and_cash_equivalents_at_carrying_value")
CURRENT_ASSETS_KEYS = ("assets_current", "current_assets")
CURRENT_LIABILITIES_KEYS = ("liabilities_current", "current_liabilities")
LIABILITIES_KEYS = ("liabilities", "total_liabilities")
EQUITY_KEYS = ("stockholders_equity", "shareholders_equity")
CURRENT_RATIO_KEYS = ("current_ratio",)

GUIDANCE_KEYWORDS = ("guidance", "outlook", "expects", "forecast", "projection")
RISK_KEYWORDS = ("risk", "risks", "uncertain", "uncertainty", "litigation", "competition")
ACCOUNTING_KEYWORDS = (
    "critical accounting",
    "accounting estimate",
    "accounting policy",
    "impairment",
    "estimate",
    "tax",
)

QUESTION_TEXT: Mapping[FinancialReportSection, str] = {
    FinancialReportSection.REVENUE: (
        "What do stored statements show about revenue and prior-period change?"
    ),
    FinancialReportSection.MARGINS: (
        "What do stored statements show about gross, operating, and net margins?"
    ),
    FinancialReportSection.CASH_FLOW: (
        "What do stored statements show about operating cash flow and free cash flow proxy?"
    ),
    FinancialReportSection.DEBT_LIQUIDITY: (
        "What do stored statements show about liquidity, liabilities, and equity?"
    ),
    FinancialReportSection.GUIDANCE: (
        "What guidance or outlook language is present in stored filing chunks?"
    ),
    FinancialReportSection.RISKS: "What risk language is present in stored filing chunks?",
    FinancialReportSection.ACCOUNTING_CAVEATS: (
        "What accounting estimate or caveat language is present in stored filing chunks?"
    ),
}


class StatementResultStore(Protocol):
    def get_result(
        self,
        *,
        cik: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> FinancialStatementResult | None: ...


class FilingResultStore(Protocol):
    def get_result(
        self,
        *,
        cik: str,
        provider: str | None = None,
        now: datetime | None = None,
    ) -> FilingIngestionResult | None: ...


@dataclass(frozen=True, slots=True)
class _EvidencePoint:
    evidence_id: str
    citation_id: str


@dataclass(frozen=True, slots=True)
class _MetricPoint:
    statement: NormalizedFinancialStatement
    key: str
    value: Decimal
    evidence_id: str
    citation_id: str


class FinancialReportAnalysisAgent:
    """Deterministic first agent for grounded report analysis over stored local data."""

    def __init__(
        self,
        *,
        statement_store: StatementResultStore,
        filing_store: FilingResultStore,
        statement_provider: str | None = None,
        filing_provider: str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._statement_store = statement_store
        self._filing_store = filing_store
        self._statement_provider = statement_provider
        self._filing_provider = filing_provider
        self._now = now or (lambda: datetime.now(UTC))

    def analyze(
        self,
        company: FinancialReportAnalysisCompany,
    ) -> FinancialReportAnalysisResult:
        created_at = _aware_now(self._now())
        limitations: list[str] = []
        warnings: list[str] = []
        statement_result = self._stored_statements(company, created_at, limitations)
        filing_result = self._stored_filings(company, created_at, limitations)
        if statement_result is not None:
            warnings.extend(statement_result.warnings)
        else:
            limitations.append(
                "No stored financial statements were available for this CIK. Fetch SEC "
                "companyfacts before relying on revenue, margins, cash flow, or liquidity."
            )
        if filing_result is not None:
            warnings.extend(filing_result.warnings)
        else:
            limitations.append(
                "No stored filing chunks were available for this CIK. Fetch SEC filings before "
                "relying on guidance, risk, or accounting caveat analysis."
            )

        evidence_builder = _EvidenceBuilder()
        findings = (
            self._revenue_finding(statement_result, evidence_builder),
            self._margins_finding(statement_result, evidence_builder),
            self._cash_flow_finding(statement_result, evidence_builder),
            self._debt_liquidity_finding(statement_result, evidence_builder),
            self._filing_language_finding(
                section=FinancialReportSection.GUIDANCE,
                title="Guidance and Outlook",
                filing_result=filing_result,
                keywords=GUIDANCE_KEYWORDS,
                evidence_builder=evidence_builder,
            ),
            self._filing_language_finding(
                section=FinancialReportSection.RISKS,
                title="Risk Language",
                filing_result=filing_result,
                keywords=RISK_KEYWORDS,
                evidence_builder=evidence_builder,
            ),
            self._filing_language_finding(
                section=FinancialReportSection.ACCOUNTING_CAVEATS,
                title="Accounting Caveats",
                filing_result=filing_result,
                keywords=ACCOUNTING_KEYWORDS,
                evidence_builder=evidence_builder,
            ),
        )
        questions = tuple(_question_from_finding(finding) for finding in findings)
        status = _analysis_status(
            statements_available=statement_result is not None,
            filings_available=filing_result is not None,
            findings=findings,
            top_level_limitations=limitations,
        )
        return FinancialReportAnalysisResult(
            id=f"financial_report_analysis_{company.cik}_{uuid4().hex}",
            company=company,
            status=status,
            created_at=created_at,
            questions=questions,
            findings=findings,
            citations=evidence_builder.citations,
            evidence=evidence_builder.evidence,
            limitations=tuple(dict.fromkeys(limitations)),
            warnings=tuple(dict.fromkeys(warnings)),
            source_summary=_source_summary(statement_result, filing_result),
            no_recommendation_notice=NO_RECOMMENDATION_NOTICE,
        )

    def _stored_statements(
        self,
        company: FinancialReportAnalysisCompany,
        now: datetime,
        limitations: list[str],
    ) -> FinancialStatementResult | None:
        try:
            return self._statement_store.get_result(
                cik=company.cik,
                provider=self._statement_provider,
                now=now,
            )
        except Exception as exc:
            limitations.append(f"Financial statement store failed: {exc}")
            return None

    def _stored_filings(
        self,
        company: FinancialReportAnalysisCompany,
        now: datetime,
        limitations: list[str],
    ) -> FilingIngestionResult | None:
        try:
            return self._filing_store.get_result(
                cik=company.cik,
                provider=self._filing_provider,
                now=now,
            )
        except Exception as exc:
            limitations.append(f"Filing store failed: {exc}")
            return None

    def _revenue_finding(
        self,
        statement_result: FinancialStatementResult | None,
        evidence_builder: _EvidenceBuilder,
    ) -> FinancialReportFinding:
        latest = _metric_point(
            statement_result,
            FinancialStatementType.INCOME_STATEMENT,
            REVENUE_KEYS,
            evidence_builder,
            FinancialReportSection.REVENUE,
        )
        previous = _metric_point(
            statement_result,
            FinancialStatementType.INCOME_STATEMENT,
            REVENUE_KEYS,
            evidence_builder,
            FinancialReportSection.REVENUE,
            skip_statement_id=latest.statement.id if latest is not None else None,
        )
        if latest is None:
            limitation = "Stored income statements do not include a supported revenue line item."
            return _limited_finding(
                FinancialReportSection.REVENUE,
                "Revenue",
                limitation,
            )

        amount = _format_amount(latest.value, latest.statement.currency)
        summary = (
            f"Stored income statement data reports {amount} in {latest.key} for fiscal "
            f"year {latest.statement.period.fiscal_year}."
        )
        comparison = _comparison(latest, previous, "Revenue")
        return FinancialReportFinding(
            id="finding:revenue",
            section=FinancialReportSection.REVENUE,
            title="Revenue",
            summary=summary,
            confidence=ConfidenceLabel.HIGH,
            evidence_ids=_evidence_ids(latest, previous),
            citation_ids=_citation_ids(latest, previous),
            prior_period_comparison=comparison,
            limitations=(
                ()
                if previous is not None
                else ("No prior-period revenue evidence was available for comparison.",)
            ),
        )

    def _margins_finding(
        self,
        statement_result: FinancialStatementResult | None,
        evidence_builder: _EvidenceBuilder,
    ) -> FinancialReportFinding:
        ratio_statements = _statements(statement_result, FinancialStatementType.KEY_RATIOS)
        ratio_points = [
            point
            for point in (
                _metric_point_from_statements(
                    ratio_statements,
                    ("gross_margin",),
                    evidence_builder,
                    FinancialReportSection.MARGINS,
                ),
                _metric_point_from_statements(
                    ratio_statements,
                    ("operating_margin",),
                    evidence_builder,
                    FinancialReportSection.MARGINS,
                ),
                _metric_point_from_statements(
                    ratio_statements,
                    ("net_margin",),
                    evidence_builder,
                    FinancialReportSection.MARGINS,
                ),
            )
            if point is not None
        ]
        if ratio_points:
            summary = (
                "Stored ratio data reports "
                + ", ".join(
                    f"{point.key} of {_format_percent(point.value)}" for point in ratio_points
                )
                + "."
            )
            return FinancialReportFinding(
                id="finding:margins",
                section=FinancialReportSection.MARGINS,
                title="Margins",
                summary=summary,
                confidence=ConfidenceLabel.HIGH,
                evidence_ids=tuple(point.evidence_id for point in ratio_points),
                citation_ids=tuple(point.citation_id for point in ratio_points),
                limitations=_margin_limitations(tuple(point.key for point in ratio_points)),
            )

        calculated = _calculated_margin_points(statement_result, evidence_builder)
        if calculated:
            summary_parts = [
                text for label, (text, _) in calculated.items() if label != "revenue_basis"
            ]
            summary = "Calculated from stored line items: " + ", ".join(summary_parts) + "."
            points = tuple(point for _, point in calculated.values())
            return FinancialReportFinding(
                id="finding:margins",
                section=FinancialReportSection.MARGINS,
                title="Margins",
                summary=summary,
                confidence=ConfidenceLabel.MEDIUM,
                evidence_ids=tuple(dict.fromkeys(point.evidence_id for point in points)),
                citation_ids=tuple(dict.fromkeys(point.citation_id for point in points)),
                limitations=("Margin ratios were calculated locally from stored line items.",),
            )

        limitation = (
            "Stored statements do not include enough revenue and income line items to assess "
            "gross, operating, or net margins."
        )
        return _limited_finding(FinancialReportSection.MARGINS, "Margins", limitation)

    def _cash_flow_finding(
        self,
        statement_result: FinancialStatementResult | None,
        evidence_builder: _EvidenceBuilder,
    ) -> FinancialReportFinding:
        ocf = _metric_point(
            statement_result,
            FinancialStatementType.CASH_FLOW,
            OPERATING_CASH_FLOW_KEYS,
            evidence_builder,
            FinancialReportSection.CASH_FLOW,
        )
        fcf_proxy = _metric_point(
            statement_result,
            FinancialStatementType.KEY_RATIOS,
            FREE_CASH_FLOW_KEYS,
            evidence_builder,
            FinancialReportSection.CASH_FLOW,
        )
        if ocf is None and fcf_proxy is None:
            limitation = (
                "Stored statements do not include operating cash flow or free cash flow proxy "
                "line items."
            )
            return _limited_finding(FinancialReportSection.CASH_FLOW, "Cash Flow", limitation)

        parts = []
        points = []
        if ocf is not None:
            points.append(ocf)
            parts.append(
                f"operating cash flow of {_format_amount(ocf.value, ocf.statement.currency)}"
            )
        if fcf_proxy is not None:
            points.append(fcf_proxy)
            amount = _format_amount(fcf_proxy.value, fcf_proxy.statement.currency)
            parts.append(f"free cash flow proxy of {amount}")
        summary = "Stored cash flow evidence reports " + " and ".join(parts) + "."
        return FinancialReportFinding(
            id="finding:cash_flow",
            section=FinancialReportSection.CASH_FLOW,
            title="Cash Flow",
            summary=summary,
            confidence=ConfidenceLabel.HIGH,
            evidence_ids=tuple(point.evidence_id for point in points),
            citation_ids=tuple(point.citation_id for point in points),
            limitations=(
                ()
                if ocf is not None and fcf_proxy is not None
                else ("Cash flow coverage is partial for the stored statement set.",)
            ),
        )

    def _debt_liquidity_finding(
        self,
        statement_result: FinancialStatementResult | None,
        evidence_builder: _EvidenceBuilder,
    ) -> FinancialReportFinding:
        current_ratio = _metric_point(
            statement_result,
            FinancialStatementType.KEY_RATIOS,
            CURRENT_RATIO_KEYS,
            evidence_builder,
            FinancialReportSection.DEBT_LIQUIDITY,
        )
        current_assets = None
        current_liabilities = None
        if current_ratio is None:
            current_assets = _metric_point(
                statement_result,
                FinancialStatementType.BALANCE_SHEET,
                CURRENT_ASSETS_KEYS,
                evidence_builder,
                FinancialReportSection.DEBT_LIQUIDITY,
            )
            current_liabilities = _metric_point(
                statement_result,
                FinancialStatementType.BALANCE_SHEET,
                CURRENT_LIABILITIES_KEYS,
                evidence_builder,
                FinancialReportSection.DEBT_LIQUIDITY,
            )
        cash = _metric_point(
            statement_result,
            FinancialStatementType.BALANCE_SHEET,
            CASH_KEYS,
            evidence_builder,
            FinancialReportSection.DEBT_LIQUIDITY,
        )
        liabilities = _metric_point(
            statement_result,
            FinancialStatementType.BALANCE_SHEET,
            LIABILITIES_KEYS,
            evidence_builder,
            FinancialReportSection.DEBT_LIQUIDITY,
        )
        equity = _metric_point(
            statement_result,
            FinancialStatementType.BALANCE_SHEET,
            EQUITY_KEYS,
            evidence_builder,
            FinancialReportSection.DEBT_LIQUIDITY,
        )
        calculated_current_ratio = (
            _safe_divide(current_assets.value, current_liabilities.value)
            if current_ratio is None
            and current_assets is not None
            and current_liabilities is not None
            else None
        )
        points = tuple(
            point
            for point in (
                current_ratio,
                current_assets,
                current_liabilities,
                cash,
                liabilities,
                equity,
            )
            if point
        )
        if not points:
            limitation = (
                "Stored balance sheet data does not include supported liquidity, liability, "
                "or equity line items."
            )
            return _limited_finding(
                FinancialReportSection.DEBT_LIQUIDITY,
                "Debt and Liquidity",
                limitation,
            )

        parts = []
        if current_ratio is not None:
            parts.append(f"current ratio {_format_decimal(current_ratio.value)}")
        elif calculated_current_ratio is not None:
            parts.append(f"calculated current ratio {_format_decimal(calculated_current_ratio)}")
        if cash is not None:
            parts.append(f"cash of {_format_amount(cash.value, cash.statement.currency)}")
        if liabilities is not None:
            amount = _format_amount(liabilities.value, liabilities.statement.currency)
            parts.append(f"liabilities of {amount}")
        if equity is not None:
            parts.append(f"equity of {_format_amount(equity.value, equity.statement.currency)}")
        return FinancialReportFinding(
            id="finding:debt_liquidity",
            section=FinancialReportSection.DEBT_LIQUIDITY,
            title="Debt and Liquidity",
            summary="Stored balance sheet and ratio evidence reports " + ", ".join(parts) + ".",
            confidence=ConfidenceLabel.HIGH,
            evidence_ids=tuple(point.evidence_id for point in points),
            citation_ids=tuple(point.citation_id for point in points),
            limitations=_debt_liquidity_limitations(
                current_ratio=current_ratio,
                calculated_current_ratio=calculated_current_ratio,
            ),
        )

    def _filing_language_finding(
        self,
        *,
        section: FinancialReportSection,
        title: str,
        filing_result: FilingIngestionResult | None,
        keywords: tuple[str, ...],
        evidence_builder: _EvidenceBuilder,
    ) -> FinancialReportFinding:
        chunks = _matching_chunks(filing_result, keywords, limit=2)
        if not chunks:
            limitation = (
                f"Stored filing chunks did not contain supported {title.lower()} keywords. "
                "This is a retrieval limitation, not evidence that the topic is absent."
            )
            return _limited_finding(section, title, limitation)

        evidence_points = tuple(
            evidence_builder.add_chunk(
                chunk,
                section,
                retrieved_at=filing_result.source.retrieved_at,
            )
            for chunk in chunks
        )
        headings = tuple(
            dict.fromkeys(
                chunk.section_heading or f"{chunk.form_type} chunk {chunk.chunk_index}"
                for chunk in chunks
            )
        )
        summary = (
            f"Stored filing text contains {title.lower()} language in "
            f"{', '.join(headings)}. Review the cited chunks before using this in a report."
        )
        return FinancialReportFinding(
            id=f"finding:{section.value}",
            section=section,
            title=title,
            summary=summary,
            confidence=ConfidenceLabel.MEDIUM,
            evidence_ids=tuple(point.evidence_id for point in evidence_points),
            citation_ids=tuple(point.citation_id for point in evidence_points),
            limitations=(
                "Filing language is keyword-selected and has not been interpreted by an LLM "
                "or analyst workflow.",
            ),
        )


class _EvidenceBuilder:
    def __init__(self) -> None:
        self._citations: list[Citation] = []
        self._evidence: list[EvidenceSnippet] = []

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(self._citations)

    @property
    def evidence(self) -> tuple[EvidenceSnippet, ...]:
        return tuple(self._evidence)

    def add_statement(
        self,
        statement: NormalizedFinancialStatement,
        key: str,
        value: Decimal,
        section: FinancialReportSection,
    ) -> _MetricPoint:
        citation_id = self._next_citation_id()
        evidence_id = f"statement:{statement.id}:{key}:{section.value}"
        text = (
            f"Financial statement line item {key} for fiscal year "
            f"{statement.period.fiscal_year}: {_format_amount(value, statement.currency)}."
        )
        metadata = {
            "kind": "financial_statement",
            "cik": statement.company.cik,
            "statement_type": statement.statement_type.value,
            "fiscal_year": str(statement.period.fiscal_year),
            "period_end": statement.period.period_end.isoformat(),
            "line_item": key,
            "currency": statement.currency,
        }
        concept = statement.source.concept_mappings.get(key)
        if concept:
            metadata["taxonomy_concept"] = concept
        citation = Citation(
            id=citation_id,
            evidence_id=evidence_id,
            source_url=statement.source.source_url,
            retrieved_at=statement.source.retrieved_at,
            source_id=statement.source.provider,
            document_id=statement.period.accession_number,
            section=section.value,
            quote=text,
            metadata=metadata,
        )
        evidence = EvidenceSnippet(
            id=evidence_id,
            citation_id=citation_id,
            text=text,
            source_url=statement.source.source_url,
            retrieved_at=statement.source.retrieved_at,
            score=1.0,
            source_id=statement.source.provider,
            document_id=statement.period.accession_number,
            section=section.value,
            metadata=metadata,
        )
        self._citations.append(citation)
        self._evidence.append(evidence)
        return _MetricPoint(statement, key, value, evidence_id, citation_id)

    def add_chunk(
        self,
        chunk: FilingChunk,
        section: FinancialReportSection,
        *,
        retrieved_at: datetime,
    ) -> _EvidencePoint:
        citation_id = self._next_citation_id()
        evidence_id = f"filing:{chunk.id}:{section.value}"
        quote = _shorten(" ".join(chunk.text.split()), 500)
        metadata = {
            "kind": "filing_chunk",
            "accession_number": chunk.accession_number,
            "form_type": chunk.form_type,
            "chunk_index": str(chunk.chunk_index),
        }
        if chunk.section_heading is not None:
            metadata["section_heading"] = chunk.section_heading
        citation = Citation(
            id=citation_id,
            evidence_id=evidence_id,
            source_url=chunk.source_url,
            retrieved_at=retrieved_at,
            document_id=chunk.filing_id,
            chunk_id=chunk.id,
            section=chunk.section_heading or section.value,
            quote=quote,
            metadata=metadata,
        )
        evidence = EvidenceSnippet(
            id=evidence_id,
            citation_id=citation_id,
            text=_shorten(" ".join(chunk.text.split()), 1_200),
            source_url=chunk.source_url,
            retrieved_at=retrieved_at,
            score=1.0,
            document_id=chunk.filing_id,
            chunk_id=chunk.id,
            section=chunk.section_heading or section.value,
            metadata=metadata,
        )
        self._citations.append(citation)
        self._evidence.append(evidence)
        return _EvidencePoint(evidence_id=evidence_id, citation_id=citation_id)

    def _next_citation_id(self) -> str:
        return f"C{len(self._citations) + 1}"


def _metric_point(
    statement_result: FinancialStatementResult | None,
    statement_type: FinancialStatementType,
    keys: tuple[str, ...],
    evidence_builder: _EvidenceBuilder,
    section: FinancialReportSection,
    *,
    skip_statement_id: str | None = None,
) -> _MetricPoint | None:
    statements = tuple(
        statement
        for statement in _statements(statement_result, statement_type)
        if statement.id != skip_statement_id
    )
    return _metric_point_from_statements(statements, keys, evidence_builder, section)


def _metric_point_from_statements(
    statements: tuple[NormalizedFinancialStatement, ...],
    keys: tuple[str, ...],
    evidence_builder: _EvidenceBuilder,
    section: FinancialReportSection,
) -> _MetricPoint | None:
    for statement in statements:
        match = _line_item(statement.line_items, keys)
        if match is not None:
            key, value = match
            return evidence_builder.add_statement(statement, key, value, section)
    return None


def _calculated_margin_points(
    statement_result: FinancialStatementResult | None,
    evidence_builder: _EvidenceBuilder,
) -> Mapping[str, tuple[str, _MetricPoint]]:
    income_statements = _statements(statement_result, FinancialStatementType.INCOME_STATEMENT)
    if not income_statements:
        return {}
    latest = income_statements[0]
    revenue = _line_item(latest.line_items, REVENUE_KEYS)
    if revenue is None or revenue[1] == 0:
        return {}
    calculated: dict[str, tuple[str, _MetricPoint]] = {}
    revenue_point = evidence_builder.add_statement(
        latest,
        revenue[0],
        revenue[1],
        FinancialReportSection.MARGINS,
    )
    for label, keys in (
        ("gross_margin", GROSS_PROFIT_KEYS),
        ("operating_margin", OPERATING_INCOME_KEYS),
        ("net_margin", NET_INCOME_KEYS),
    ):
        metric = _line_item(latest.line_items, keys)
        if metric is None:
            continue
        metric_point = evidence_builder.add_statement(
            latest,
            metric[0],
            metric[1],
            FinancialReportSection.MARGINS,
        )
        ratio = _safe_divide(metric[1], revenue[1])
        if ratio is not None:
            calculated[label] = (f"{label} {_format_percent(ratio)}", metric_point)
    if calculated:
        calculated["revenue_basis"] = ("revenue basis", revenue_point)
    return calculated


def _statements(
    result: FinancialStatementResult | None,
    statement_type: FinancialStatementType,
) -> tuple[NormalizedFinancialStatement, ...]:
    if result is None:
        return ()
    return tuple(
        sorted(
            (
                statement
                for statement in result.statements
                if statement.statement_type == statement_type
            ),
            key=lambda statement: (
                statement.period.period_end,
                statement.period.fiscal_year,
                statement.id,
            ),
            reverse=True,
        )
    )


def _line_item(
    line_items: Mapping[str, Decimal],
    keys: tuple[str, ...],
) -> tuple[str, Decimal] | None:
    for key in keys:
        if key in line_items:
            return key, line_items[key]
    return None


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == 0:
        return None
    try:
        return numerator / denominator
    except InvalidOperation, ZeroDivisionError:
        return None


def _matching_chunks(
    filing_result: FilingIngestionResult | None,
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> tuple[FilingChunk, ...]:
    if filing_result is None:
        return ()
    scored: list[tuple[int, FilingChunk]] = []
    for chunk in filing_result.chunks:
        lowered = chunk.text.lower()
        score = sum(lowered.count(keyword) for keyword in keywords)
        if score > 0:
            scored.append((score, chunk))
    return tuple(
        chunk
        for _, chunk in sorted(
            scored,
            key=lambda item: (item[0], -item[1].chunk_index),
            reverse=True,
        )[:limit]
    )


def _question_from_finding(finding: FinancialReportFinding) -> FinancialReportQuestion:
    return FinancialReportQuestion(
        id=f"question:{finding.section.value}",
        section=finding.section,
        question=QUESTION_TEXT[finding.section],
        evidence_ids=finding.evidence_ids,
        limitations=finding.limitations,
    )


def _limited_finding(
    section: FinancialReportSection,
    title: str,
    limitation: str,
) -> FinancialReportFinding:
    return FinancialReportFinding(
        id=f"finding:{section.value}",
        section=section,
        title=title,
        summary=limitation,
        confidence=ConfidenceLabel.UNKNOWN,
        limitations=(limitation,),
    )


def _margin_limitations(keys: tuple[str, ...]) -> tuple[str, ...]:
    missing = [
        label for label in ("gross_margin", "operating_margin", "net_margin") if label not in keys
    ]
    if not missing:
        return ()
    return (f"Stored ratio data did not include: {', '.join(missing)}.",)


def _debt_liquidity_limitations(
    *,
    current_ratio: _MetricPoint | None,
    calculated_current_ratio: Decimal | None,
) -> tuple[str, ...]:
    limitations = ["Debt-specific line items are only assessed when present in stored statements."]
    if current_ratio is None and calculated_current_ratio is not None:
        limitations.append(
            "Current ratio was calculated locally from stored current assets and current "
            "liabilities."
        )
    elif current_ratio is None:
        limitations.append("Stored data did not include enough evidence for current ratio.")
    return tuple(limitations)


def _comparison(
    latest: _MetricPoint,
    previous: _MetricPoint | None,
    label: str,
) -> str | None:
    if previous is None:
        return None
    change = latest.value - previous.value
    percent = _safe_divide(change, abs(previous.value)) if previous.value != 0 else None
    direction = "increased" if change >= 0 else "decreased"
    base = (
        f"{label} {direction} from {_format_amount(previous.value, previous.statement.currency)} "
        f"in fiscal year {previous.statement.period.fiscal_year} to "
        f"{_format_amount(latest.value, latest.statement.currency)} in fiscal year "
        f"{latest.statement.period.fiscal_year}."
    )
    if percent is None:
        return base
    return f"{base} Change: {_format_percent(percent)}."


def _analysis_status(
    *,
    statements_available: bool,
    filings_available: bool,
    findings: tuple[FinancialReportFinding, ...],
    top_level_limitations: Iterable[str],
) -> FinancialReportAnalysisStatus:
    if not statements_available and not filings_available:
        return FinancialReportAnalysisStatus.NO_DATA
    if any(finding.limitations for finding in findings) or tuple(top_level_limitations):
        return FinancialReportAnalysisStatus.PARTIAL
    return FinancialReportAnalysisStatus.COMPLETE


def _source_summary(
    statement_result: FinancialStatementResult | None,
    filing_result: FilingIngestionResult | None,
) -> Mapping[str, str]:
    summary = {
        "statement_count": str(len(statement_result.statements) if statement_result else 0),
        "filing_count": str(len(filing_result.filings) if filing_result else 0),
        "filing_chunk_count": str(len(filing_result.chunks) if filing_result else 0),
    }
    if statement_result is not None:
        summary["statement_provider"] = statement_result.source.provider
        if statement_result.source.data_as_of is not None:
            summary["statement_data_as_of"] = statement_result.source.data_as_of.isoformat()
    if filing_result is not None:
        summary["filing_provider"] = filing_result.source.provider
        if filing_result.source.data_as_of is not None:
            summary["filing_data_as_of"] = filing_result.source.data_as_of.isoformat()
    return summary


def _evidence_ids(*points: _MetricPoint | None) -> tuple[str, ...]:
    return tuple(point.evidence_id for point in points if point is not None)


def _citation_ids(*points: _MetricPoint | None) -> tuple[str, ...]:
    return tuple(point.citation_id for point in points if point is not None)


def _format_amount(value: Decimal, currency: str) -> str:
    return f"{_format_decimal(value)} {currency.upper()}"


def _format_percent(value: Decimal) -> str:
    return f"{(value * Decimal('100')).quantize(Decimal('0.1'))}%"


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return format(value.quantize(Decimal("1")), "f")
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _aware_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
