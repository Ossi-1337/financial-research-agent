from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.context_analysis import (
    ContextAnalysisResult,
    ContextAnalysisStatus,
    ContextScope,
    NewsMacroSectorAgent,
)
from financial_research_agent.entities import (
    CompanySearchCandidate,
    CompanySearchError,
    CompanySearchProvider,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifierType,
    ResolvedSecurity,
)
from financial_research_agent.filings import (
    FilingCompany,
    FilingError,
    FilingProvider,
    FilingStore,
)
from financial_research_agent.market_data import (
    MarketDataError,
    MarketDataProvider,
    MarketDataStore,
    MarketSecurity,
)
from financial_research_agent.orchestration.contracts import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.orchestration.store import OrchestratorRunStore
from financial_research_agent.report_analysis import (
    FinancialReportAnalysisAgent,
    FinancialReportAnalysisCompany,
    FinancialReportAnalysisStatus,
)
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementProvider,
    FinancialStatementStore,
)
from financial_research_agent.stock_analysis import (
    StockPriceAnalysisAgent,
    StockPriceAnalysisSecurity,
    StockPriceAnalysisStatus,
)
from financial_research_agent.synthesis import SynthesisAgent, SynthesisReportStatus


class ResearchOrchestrator:
    """Coordinates bounded specialist workflows through declared provider/store boundaries."""

    def __init__(
        self,
        *,
        company_search_provider: CompanySearchProvider,
        market_data_provider: MarketDataProvider,
        market_data_store: MarketDataStore,
        financial_statement_provider: FinancialStatementProvider,
        financial_statement_store: FinancialStatementStore,
        filing_provider: FilingProvider,
        filing_store: FilingStore,
        financial_report_agent: FinancialReportAnalysisAgent,
        stock_price_agent: StockPriceAnalysisAgent,
        context_agent: NewsMacroSectorAgent,
        synthesis_agent: SynthesisAgent | None = None,
        run_store: OrchestratorRunStore | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._company_search_provider = company_search_provider
        self._market_data_provider = market_data_provider
        self._market_data_store = market_data_store
        self._financial_statement_provider = financial_statement_provider
        self._financial_statement_store = financial_statement_store
        self._filing_provider = filing_provider
        self._filing_store = filing_store
        self._financial_report_agent = financial_report_agent
        self._stock_price_agent = stock_price_agent
        self._context_agent = context_agent
        self._synthesis_agent = synthesis_agent or SynthesisAgent()
        self._run_store = run_store
        self._now = now or (lambda: datetime.now(UTC))

    async def run(self, request: OrchestratorResearchInput) -> OrchestratedResearchRun:
        created_at = _aware_now(self._now())
        run = OrchestratedResearchRun(
            id=f"orchestrator_run_{uuid4().hex}",
            query=request.query,
            status=OrchestratorRunStatus.RUNNING,
            created_at=created_at,
            updated_at=created_at,
            execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
            plan=default_orchestrator_plan(),
            warnings=(
                "Execution policy is sequential_local_safe to avoid overloading local model "
                "and provider resources.",
            ),
        )
        run = self._save(run)

        resolution_handoff, search_result, candidate = await self._resolve_company(request)
        run = self._append_handoff(run, resolution_handoff)
        if candidate is None:
            run = self._finish_without_candidate(run, search_result)
            return self._save(run)

        company = candidate.company
        security = candidate.securities[0]
        cik = _candidate_identifier(candidate, EntityIdentifierType.CIK)
        run = self._save(
            replace(
                run,
                selected_company=company.to_dict(),
                selected_security=security.to_dict(),
                updated_at=_aware_now(self._now()),
            )
        )

        for handoff in await self._refresh_data(request, candidate, security, cik):
            run = self._append_handoff(run, handoff)

        for handoff in self._run_specialists(request, candidate, security, cik):
            run = self._append_handoff(run, handoff)

        synthesis = self._synthesize(run)
        run = self._append_handoff(run, synthesis)
        handoffs = run.handoffs
        run = self._save(
            replace(
                run,
                status=_final_status(handoffs),
                synthesis_summary=str(synthesis.output["summary"]),
                limitations=tuple(
                    dict.fromkeys(
                        limitation for handoff in handoffs for limitation in handoff.limitations
                    )
                ),
                updated_at=_aware_now(self._now()),
            )
        )
        return run

    async def _resolve_company(
        self,
        request: OrchestratorResearchInput,
    ) -> tuple[AgentHandoff, CompanySearchResult | None, CompanySearchCandidate | None]:
        started_at = _aware_now(self._now())
        try:
            result = await self._company_search_provider.search(
                request.query,
                limit=request.company_search_limit,
            )
        except CompanySearchError as exc:
            return (
                _failed_handoff(
                    kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                    step_id="resolve_company",
                    started_at=started_at,
                    completed_at=_aware_now(self._now()),
                    input_summary={"query": request.query},
                    error_code=exc.code.value,
                    error_message=exc.message,
                ),
                None,
                None,
            )

        if result.status == CompanySearchStatus.NO_MATCHES or not result.candidates:
            limitation = "Company resolution returned no reviewable candidates."
            return (
                _handoff(
                    kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                    step_id="resolve_company",
                    status=OrchestratorHandoffStatus.FAILED,
                    started_at=started_at,
                    completed_at=_aware_now(self._now()),
                    input_summary={"query": request.query},
                    output=result.to_dict(),
                    limitations=(limitation,),
                    confidence=HandoffConfidence.UNKNOWN,
                    error_code="no_company_match",
                    error_message=limitation,
                ),
                result,
                None,
            )

        candidate = result.candidates[0]
        warnings = tuple(
            dict.fromkeys(
                (
                    *result.warnings,
                    *candidate.warnings,
                    (
                        "Top reviewable company candidate selected automatically for this "
                        "bounded workflow."
                    ),
                )
            )
        )
        return (
            _handoff(
                kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                step_id="resolve_company",
                status=OrchestratorHandoffStatus.SUCCEEDED,
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"query": request.query},
                output={"result": result.to_dict(), "selected_candidate": candidate.to_dict()},
                warnings=warnings,
                confidence=HandoffConfidence.MEDIUM,
            ),
            result,
            candidate,
        )

    async def _refresh_data(
        self,
        request: OrchestratorResearchInput,
        candidate: CompanySearchCandidate,
        security: ResolvedSecurity,
        cik: str | None,
    ) -> tuple[AgentHandoff, ...]:
        if not request.refresh:
            skipped_at = _aware_now(self._now())
            return (
                _skipped_handoff(
                    kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
                    step_id="refresh_market_data",
                    reason="Refresh disabled by request.",
                    occurred_at=skipped_at,
                ),
                _skipped_handoff(
                    kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
                    step_id="refresh_financial_statements",
                    reason="Refresh disabled by request.",
                    occurred_at=skipped_at,
                ),
                _skipped_handoff(
                    kind=OrchestratorStepKind.FILING_REFRESH,
                    step_id="refresh_filings",
                    reason="Refresh disabled by request.",
                    occurred_at=skipped_at,
                ),
            )

        handoffs = [
            await self._refresh_market_data(security, request.market_outputsize),
            await self._refresh_financial_statements(candidate, cik, request.fiscal_years),
            await self._refresh_filings(candidate, cik, request.filing_forms, request.filing_limit),
        ]
        return tuple(handoffs)

    async def _refresh_market_data(
        self,
        security: ResolvedSecurity,
        outputsize: str,
    ) -> AgentHandoff:
        started_at = _aware_now(self._now())
        market_security = MarketSecurity(
            symbol=security.ticker,
            security_id=security.id,
            exchange_mic=security.exchange_mic,
            exchange_name=security.exchange_name,
            currency=security.currency,
        )
        try:
            history = await self._market_data_provider.fetch_daily_prices(
                market_security,
                outputsize=outputsize,
            )
            stored = self._market_data_store.save_history(history)
        except MarketDataError as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
                step_id="refresh_market_data",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"symbol": market_security.symbol, "outputsize": outputsize},
                error_code=exc.code.value,
                error_message=exc.message,
            )
        return _handoff(
            kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
            step_id="refresh_market_data",
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"symbol": market_security.symbol, "outputsize": outputsize},
            output={"history": stored.to_dict()},
            warnings=stored.warnings,
            confidence=HandoffConfidence.HIGH,
        )

    async def _refresh_financial_statements(
        self,
        candidate: CompanySearchCandidate,
        cik: str | None,
        fiscal_years: int,
    ) -> AgentHandoff:
        if cik is None:
            return _skipped_handoff(
                kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
                step_id="refresh_financial_statements",
                reason="No CIK was available from company resolution.",
                occurred_at=_aware_now(self._now()),
            )
        started_at = _aware_now(self._now())
        company = FinancialStatementCompany(
            cik=cik,
            company_id=candidate.company.id,
            legal_name=candidate.company.legal_name,
        )
        try:
            result = await self._financial_statement_provider.fetch_statements(
                company,
                fiscal_years=fiscal_years,
            )
            stored = self._financial_statement_store.save_result(result)
        except FinancialStatementError as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
                step_id="refresh_financial_statements",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"cik": cik, "fiscal_years": str(fiscal_years)},
                error_code=exc.code.value,
                error_message=exc.message,
            )
        return _handoff(
            kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
            step_id="refresh_financial_statements",
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": cik, "fiscal_years": str(fiscal_years)},
            output={"statements": stored.to_dict()},
            warnings=stored.warnings,
            confidence=HandoffConfidence.HIGH,
        )

    async def _refresh_filings(
        self,
        candidate: CompanySearchCandidate,
        cik: str | None,
        forms: tuple[str, ...],
        limit: int,
    ) -> AgentHandoff:
        if cik is None:
            return _skipped_handoff(
                kind=OrchestratorStepKind.FILING_REFRESH,
                step_id="refresh_filings",
                reason="No CIK was available from company resolution.",
                occurred_at=_aware_now(self._now()),
            )
        started_at = _aware_now(self._now())
        company = FilingCompany(
            cik=cik,
            company_id=candidate.company.id,
            legal_name=candidate.company.legal_name,
        )
        try:
            result = await self._filing_provider.ingest_latest(
                company,
                forms=forms,
                limit=limit,
            )
            stored = self._filing_store.save_result(result)
        except FilingError as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.FILING_REFRESH,
                step_id="refresh_filings",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"cik": cik, "forms": ",".join(forms), "limit": str(limit)},
                error_code=exc.code.value,
                error_message=exc.message,
            )
        return _handoff(
            kind=OrchestratorStepKind.FILING_REFRESH,
            step_id="refresh_filings",
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": cik, "forms": ",".join(forms), "limit": str(limit)},
            output={"filings": stored.to_dict()},
            warnings=stored.warnings,
            confidence=HandoffConfidence.HIGH,
        )

    def _run_specialists(
        self,
        request: OrchestratorResearchInput,
        candidate: CompanySearchCandidate,
        security: ResolvedSecurity,
        cik: str | None,
    ) -> tuple[AgentHandoff, ...]:
        return (
            self._run_financial_report_analysis(candidate, cik),
            self._run_stock_price_analysis(security, request.benchmark_symbol),
            self._run_context_analysis(request, security),
        )

    def _run_financial_report_analysis(
        self,
        candidate: CompanySearchCandidate,
        cik: str | None,
    ) -> AgentHandoff:
        if cik is None:
            return _skipped_handoff(
                kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                step_id="financial_report_analysis",
                reason="No CIK was available for financial report analysis.",
                occurred_at=_aware_now(self._now()),
            )
        started_at = _aware_now(self._now())
        try:
            result = self._financial_report_agent.analyze(
                FinancialReportAnalysisCompany(
                    cik=cik,
                    company_id=candidate.company.id,
                    legal_name=candidate.company.legal_name,
                )
            )
        except Exception as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                step_id="financial_report_analysis",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"cik": cik},
                error_code="financial_report_analysis_failed",
                error_message=str(exc),
            )
        return _handoff(
            kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            step_id="financial_report_analysis",
            status=_report_handoff_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": cik},
            output={"analysis": result.to_dict()},
            evidence_ids=tuple(snippet.id for snippet in result.evidence),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_report_confidence(result.status),
        )

    def _run_stock_price_analysis(
        self,
        security: ResolvedSecurity,
        benchmark_symbol: str | None,
    ) -> AgentHandoff:
        started_at = _aware_now(self._now())
        try:
            result = self._stock_price_agent.analyze(
                StockPriceAnalysisSecurity(
                    symbol=security.ticker,
                    security_id=security.id,
                    exchange_mic=security.exchange_mic,
                    exchange_name=security.exchange_name,
                    currency=security.currency,
                ),
                benchmark_symbol=benchmark_symbol,
            )
        except Exception as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
                step_id="stock_price_analysis",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"symbol": security.ticker},
                error_code="stock_price_analysis_failed",
                error_message=str(exc),
            )
        return _handoff(
            kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            step_id="stock_price_analysis",
            status=_stock_handoff_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"symbol": security.ticker},
            output={"analysis": result.to_dict()},
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_stock_confidence(result.status),
        )

    def _run_context_analysis(
        self,
        request: OrchestratorResearchInput,
        security: ResolvedSecurity,
    ) -> AgentHandoff:
        started_at = _aware_now(self._now())
        try:
            result = self._context_agent.analyze(
                query=request.query,
                source_items=request.context_source_items,
                company_symbols=(security.ticker,),
            )
        except Exception as exc:
            return _failed_handoff(
                kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
                step_id="context_analysis",
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"source_item_count": str(len(request.context_source_items))},
                error_code="context_analysis_failed",
                error_message=str(exc),
            )
        return _handoff(
            kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
            step_id="context_analysis",
            status=_context_handoff_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"source_item_count": str(len(request.context_source_items))},
            output={"analysis": result.to_dict()},
            evidence_ids=_context_evidence_ids(result),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_context_confidence(result.status),
        )

    def _synthesize(self, run: OrchestratedResearchRun) -> AgentHandoff:
        started_at = _aware_now(self._now())
        specialist_handoffs = tuple(
            handoff
            for handoff in run.handoffs
            if handoff.kind
            in {
                OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
                OrchestratorStepKind.CONTEXT_ANALYSIS,
            }
        )
        report = self._synthesis_agent.synthesize(
            query=run.query,
            handoffs=run.handoffs,
            selected_company=run.selected_company,
            selected_security=run.selected_security,
            created_at=started_at,
        )
        report_unknown_limitations = tuple(
            limitation for point in report.unknowns for limitation in point.limitations
        )
        limitations = tuple(
            dict.fromkeys(
                (
                    *report.limitations,
                    *report_unknown_limitations,
                )
            )
        )
        specialist_warnings = tuple(
            warning for handoff in specialist_handoffs for warning in handoff.warnings
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *report.warnings,
                    *specialist_warnings,
                )
            )
        )
        return _handoff(
            kind=OrchestratorStepKind.SYNTHESIS,
            step_id="synthesis",
            status=(
                OrchestratorHandoffStatus.SUCCEEDED
                if report.status == SynthesisReportStatus.COMPLETE
                else OrchestratorHandoffStatus.PARTIAL
            ),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"specialist_handoff_count": str(len(specialist_handoffs))},
            output={
                "summary": report.summary,
                "report": report.to_dict(),
                "specialist_statuses": {
                    handoff.kind.value: handoff.status.value for handoff in specialist_handoffs
                },
            },
            evidence_ids=report.evidence_ids,
            warnings=warnings,
            limitations=limitations,
            confidence=(
                HandoffConfidence.MEDIUM if specialist_handoffs else HandoffConfidence.UNKNOWN
            ),
        )

    def _append_handoff(
        self,
        run: OrchestratedResearchRun,
        handoff: AgentHandoff,
    ) -> OrchestratedResearchRun:
        return self._save(
            replace(
                run,
                handoffs=(*run.handoffs, handoff),
                updated_at=_aware_now(self._now()),
            )
        )

    def _finish_without_candidate(
        self,
        run: OrchestratedResearchRun,
        search_result: CompanySearchResult | None,
    ) -> OrchestratedResearchRun:
        limitation = "Workflow stopped because no company candidate could be selected."
        return replace(
            run,
            status=OrchestratorRunStatus.FAILED,
            synthesis_summary=limitation,
            limitations=tuple(dict.fromkeys((*run.limitations, limitation))),
            warnings=(
                tuple(dict.fromkeys((*run.warnings, *search_result.warnings)))
                if search_result is not None
                else run.warnings
            ),
            updated_at=_aware_now(self._now()),
        )

    def _save(self, run: OrchestratedResearchRun) -> OrchestratedResearchRun:
        if self._run_store is None:
            return run
        return self._run_store.save(run)


def _handoff(
    *,
    kind: OrchestratorStepKind,
    step_id: str,
    status: OrchestratorHandoffStatus,
    started_at: datetime,
    completed_at: datetime,
    input_summary: dict[str, str] | None = None,
    output: dict[str, object] | None = None,
    evidence_ids: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    confidence: HandoffConfidence = HandoffConfidence.UNKNOWN,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff_{kind.value}_{uuid4().hex}",
        step_id=step_id,
        kind=kind,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        input_summary=input_summary or {},
        output=output or {},
        evidence_ids=evidence_ids,
        warnings=warnings,
        limitations=limitations,
        confidence=confidence,
        error_code=error_code,
        error_message=error_message,
    )


def _failed_handoff(
    *,
    kind: OrchestratorStepKind,
    step_id: str,
    started_at: datetime,
    completed_at: datetime,
    input_summary: dict[str, str],
    error_code: str,
    error_message: str,
) -> AgentHandoff:
    return _handoff(
        kind=kind,
        step_id=step_id,
        status=OrchestratorHandoffStatus.FAILED,
        started_at=started_at,
        completed_at=completed_at,
        input_summary=input_summary,
        limitations=(error_message,),
        confidence=HandoffConfidence.UNKNOWN,
        error_code=error_code,
        error_message=error_message,
    )


def _skipped_handoff(
    *,
    kind: OrchestratorStepKind,
    step_id: str,
    reason: str,
    occurred_at: datetime | None = None,
) -> AgentHandoff:
    now = _aware_now(occurred_at or datetime.now(UTC))
    return _handoff(
        kind=kind,
        step_id=step_id,
        status=OrchestratorHandoffStatus.SKIPPED,
        started_at=now,
        completed_at=now,
        limitations=(reason,),
        confidence=HandoffConfidence.UNKNOWN,
    )


def _candidate_identifier(
    candidate: CompanySearchCandidate,
    identifier_type: EntityIdentifierType,
) -> str | None:
    for identifier in candidate.company.identifiers:
        if identifier.identifier_type == identifier_type:
            return identifier.value
    for security in candidate.securities:
        for identifier in security.identifiers:
            if identifier.identifier_type == identifier_type:
                return identifier.value
    return None


def _report_handoff_status(
    status: FinancialReportAnalysisStatus,
) -> OrchestratorHandoffStatus:
    if status == FinancialReportAnalysisStatus.COMPLETE:
        return OrchestratorHandoffStatus.SUCCEEDED
    return OrchestratorHandoffStatus.PARTIAL


def _stock_handoff_status(status: StockPriceAnalysisStatus) -> OrchestratorHandoffStatus:
    if status == StockPriceAnalysisStatus.COMPLETE:
        return OrchestratorHandoffStatus.SUCCEEDED
    return OrchestratorHandoffStatus.PARTIAL


def _context_handoff_status(status: ContextAnalysisStatus) -> OrchestratorHandoffStatus:
    if status == ContextAnalysisStatus.COMPLETE:
        return OrchestratorHandoffStatus.SUCCEEDED
    return OrchestratorHandoffStatus.PARTIAL


def _report_confidence(status: FinancialReportAnalysisStatus) -> HandoffConfidence:
    if status == FinancialReportAnalysisStatus.COMPLETE:
        return HandoffConfidence.HIGH
    if status == FinancialReportAnalysisStatus.PARTIAL:
        return HandoffConfidence.MEDIUM
    return HandoffConfidence.UNKNOWN


def _stock_confidence(status: StockPriceAnalysisStatus) -> HandoffConfidence:
    if status == StockPriceAnalysisStatus.COMPLETE:
        return HandoffConfidence.HIGH
    if status == StockPriceAnalysisStatus.PARTIAL:
        return HandoffConfidence.MEDIUM
    return HandoffConfidence.UNKNOWN


def _context_confidence(status: ContextAnalysisStatus) -> HandoffConfidence:
    if status == ContextAnalysisStatus.COMPLETE:
        return HandoffConfidence.MEDIUM
    if status == ContextAnalysisStatus.PARTIAL:
        return HandoffConfidence.LOW
    return HandoffConfidence.UNKNOWN


def _context_evidence_ids(result: ContextAnalysisResult) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source_item_id
            for finding in result.findings
            for source_item_id in finding.source_item_ids
            if finding.scope in {ContextScope.COMPANY, ContextScope.MACRO, ContextScope.SECTOR}
        )
    )


def _synthesis_summary(handoffs: tuple[AgentHandoff, ...]) -> str:
    if not handoffs:
        return "No specialist outputs were available for synthesis."
    status_text = ", ".join(f"{handoff.kind.value}={handoff.status.value}" for handoff in handoffs)
    return (
        "Bounded research workflow completed from stored specialist outputs. "
        f"Specialist statuses: {status_text}. Review each handoff, evidence, warnings, "
        "and limitations before using the result."
    )


def _final_status(handoffs: tuple[AgentHandoff, ...]) -> OrchestratorRunStatus:
    if any(
        handoff.kind == OrchestratorStepKind.COMPANY_RESOLUTION
        and handoff.status == OrchestratorHandoffStatus.FAILED
        for handoff in handoffs
    ):
        return OrchestratorRunStatus.FAILED
    if any(
        handoff.status
        in {
            OrchestratorHandoffStatus.PARTIAL,
            OrchestratorHandoffStatus.SKIPPED,
            OrchestratorHandoffStatus.FAILED,
        }
        for handoff in handoffs
    ):
        return OrchestratorRunStatus.PARTIAL
    return OrchestratorRunStatus.COMPLETE


def _aware_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
