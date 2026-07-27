from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.context_analysis import (
    ContextAnalysisStatus,
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    NewsMacroSectorAgent,
    SourceReliability,
)
from financial_research_agent.filings import (
    FilingCompany,
    FilingError,
    FilingIngestionResult,
    FilingProvider,
    FilingStore,
)
from financial_research_agent.market_data import (
    MarketDataError,
    MarketDataProvider,
    MarketDataStore,
    MarketSecurity,
)
from financial_research_agent.orchestration import (
    AgentHandoff,
    AgentRole,
    DelegationRequest,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorHandoffStatus,
    OrchestratorStepKind,
)
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


class SpecialistExecutionService:
    def __init__(
        self,
        *,
        financial_report_agent: FinancialReportAnalysisAgent,
        stock_price_agent: StockPriceAnalysisAgent,
        context_agent: NewsMacroSectorAgent,
        market_data_provider: MarketDataProvider | None = None,
        market_data_store: MarketDataStore | None = None,
        financial_statement_provider: FinancialStatementProvider | None = None,
        financial_statement_store: FinancialStatementStore | None = None,
        filing_provider: FilingProvider | None = None,
        filing_store: FilingStore | None = None,
        synthesis_agent: SynthesisAgent | None = None,
        run_store: object | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.financial_report_agent = financial_report_agent
        self.stock_price_agent = stock_price_agent
        self.context_agent = context_agent
        self.market_data_provider = market_data_provider
        self.market_data_store = market_data_store
        self.financial_statement_provider = financial_statement_provider
        self.financial_statement_store = financial_statement_store
        self.filing_provider = filing_provider
        self.filing_store = filing_store
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.run_store = run_store
        self._now = now or (lambda: datetime.now(UTC))

    async def execute(self, request: DelegationRequest) -> AgentHandoff:
        if request.expected_kind == OrchestratorStepKind.MARKET_DATA_REFRESH:
            return await self._refresh_market_data(request)
        if request.expected_kind == OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH:
            return await self._refresh_financial_statements(request)
        if request.expected_kind == OrchestratorStepKind.FILING_REFRESH:
            return await self._refresh_filings(request)
        if request.role == AgentRole.FINANCIAL_REPORT:
            return self._financial_report(request)
        if request.role == AgentRole.STOCK:
            return self._stock(request)
        if request.role == AgentRole.CONTEXT:
            return self._context(request)
        if request.role == AgentRole.SYNTHESIS:
            return self._synthesis(request)
        raise ValueError(f"unsupported specialist role: {request.role.value}")

    async def _refresh_market_data(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.STOCK:
            raise ValueError("market data refresh requires stock specialist")
        provider = _required_dependency(self.market_data_provider, "market_data_provider")
        store = _required_dependency(self.market_data_store, "market_data_store")
        security = MarketSecurity(
            symbol=_text(request.payload, "ticker"),
            security_id=_optional_text(request.payload.get("security_id")),
            exchange_mic=_optional_text(request.payload.get("exchange_mic")),
            exchange_name=_optional_text(request.payload.get("exchange_name")),
            currency=_optional_text(request.payload.get("currency")),
        )
        outputsize = _text(request.payload, "outputsize")
        benchmark_symbol = _optional_text(request.payload.get("benchmark_symbol"))
        try:
            history = await provider.fetch_daily_prices(security, outputsize=outputsize)
            stored = store.save_history(history)
        except MarketDataError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)

        output: dict[str, object] = {"history": stored.to_dict()}
        warnings = list(stored.warnings)
        limitations: list[str] = []
        status = OrchestratorHandoffStatus.SUCCEEDED
        error_code = None
        error_message = None
        if benchmark_symbol is not None and benchmark_symbol != security.symbol:
            try:
                benchmark = await provider.fetch_daily_prices(
                    MarketSecurity(
                        symbol=benchmark_symbol,
                        security_id=f"benchmark:{benchmark_symbol}",
                    ),
                    outputsize=outputsize,
                )
                stored_benchmark = store.save_history(benchmark)
                output["benchmark_history"] = stored_benchmark.to_dict()
                warnings.extend(stored_benchmark.warnings)
            except MarketDataError as exc:
                status = OrchestratorHandoffStatus.PARTIAL
                error_code = exc.code.value
                error_message = exc.message
                limitations.append(
                    f"Benchmark {benchmark_symbol} could not be refreshed: {exc.message}"
                )
        return _handoff(
            request=request,
            status=status,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={
                "symbol": security.symbol,
                "outputsize": outputsize,
                **({"benchmark_symbol": benchmark_symbol} if benchmark_symbol else {}),
            },
            output=output,
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=tuple(limitations),
            confidence=HandoffConfidence.HIGH,
            error_code=error_code,
            error_message=error_message,
        )

    async def _refresh_financial_statements(
        self,
        request: DelegationRequest,
    ) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.FINANCIAL_REPORT:
            raise ValueError("statement refresh requires financial-report specialist")
        cik = _optional_text(request.payload.get("cik"))
        if cik is None:
            return _skipped_handoff(
                request,
                started_at,
                "No CIK was available from company resolution.",
            )
        provider = _required_dependency(
            self.financial_statement_provider,
            "financial_statement_provider",
        )
        store = _required_dependency(
            self.financial_statement_store,
            "financial_statement_store",
        )
        fiscal_years = _positive_int(request.payload, "fiscal_years")
        company = FinancialStatementCompany(
            cik=cik,
            company_id=_optional_text(request.payload.get("company_id")),
            legal_name=_optional_text(request.payload.get("legal_name")),
        )
        try:
            result = await provider.fetch_statements(company, fiscal_years=fiscal_years)
            stored = store.save_result(result)
        except FinancialStatementError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)
        return _handoff(
            request=request,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": cik, "fiscal_years": str(fiscal_years)},
            output={"statements": stored.to_dict()},
            warnings=stored.warnings,
            confidence=HandoffConfidence.HIGH,
        )

    async def _refresh_filings(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.FINANCIAL_REPORT:
            raise ValueError("filing refresh requires financial-report specialist")
        cik = _optional_text(request.payload.get("cik"))
        if cik is None:
            return _skipped_handoff(
                request,
                started_at,
                "No CIK was available from company resolution.",
            )
        provider = _required_dependency(self.filing_provider, "filing_provider")
        store = _required_dependency(self.filing_store, "filing_store")
        forms = _text_tuple(request.payload, "forms")
        limit = _positive_int(request.payload, "limit")
        form_limits = _positive_int_mapping(request.payload.get("form_limits", {}))
        company = FilingCompany(
            cik=cik,
            company_id=_optional_text(request.payload.get("company_id")),
            legal_name=_optional_text(request.payload.get("legal_name")),
        )
        try:
            result, partial_errors = await _ingest_filings(
                provider,
                company,
                forms=forms,
                limit=limit,
                form_limits=form_limits,
            )
            stored = store.save_result(result)
        except FilingError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)
        return _handoff(
            request=request,
            status=(
                OrchestratorHandoffStatus.PARTIAL
                if partial_errors
                else OrchestratorHandoffStatus.SUCCEEDED
            ),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={
                "cik": cik,
                "forms": ",".join(forms),
                "limit": str(limit),
            },
            output={"filings": stored.to_dict()},
            warnings=stored.warnings,
            limitations=partial_errors,
            confidence=HandoffConfidence.HIGH,
        )

    def _financial_report(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        try:
            result = self.financial_report_agent.analyze(
                FinancialReportAnalysisCompany(
                    cik=_text(request.payload, "cik"),
                    company_id=_optional_text(request.payload.get("company_id")),
                    legal_name=_optional_text(request.payload.get("legal_name")),
                )
            )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": _text(request.payload, "cik")},
            output={"analysis": result.to_dict()},
            evidence_ids=tuple(snippet.id for snippet in result.evidence),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
        )

    def _stock(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        symbol = _text(request.payload, "ticker")
        try:
            result = self.stock_price_agent.analyze(
                StockPriceAnalysisSecurity(
                    symbol=symbol,
                    security_id=_optional_text(request.payload.get("security_id")),
                    exchange_mic=_optional_text(request.payload.get("exchange_mic")),
                    exchange_name=_optional_text(request.payload.get("exchange_name")),
                    currency=_optional_text(request.payload.get("currency")),
                ),
                benchmark_symbol=_optional_text(request.payload.get("benchmark_symbol")),
            )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"symbol": symbol},
            output={"analysis": result.to_dict()},
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
        )

    def _context(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        source_items = tuple(
            _context_source_from_dict(item)
            for item in _mapping_list(request.payload.get("source_items", ()))
        )
        symbols = tuple(str(item) for item in _list(request.payload.get("company_symbols", ())))
        try:
            result = self.context_agent.analyze(
                query=_text(request.payload, "query"),
                source_items=source_items,
                company_symbols=symbols,
            )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        evidence_ids = tuple(
            source_item_id
            for finding in result.findings
            for source_item_id in finding.source_item_ids
        )
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"source_item_count": str(len(source_items))},
            output={"analysis": result.to_dict()},
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
        )

    def _synthesis(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        run = self._load_run(request.run_id)
        report = self.synthesis_agent.synthesize(
            query=run.query,
            handoffs=run.handoffs,
            selected_company=run.selected_company,
            selected_security=run.selected_security,
            created_at=started_at,
        )
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
        unknown_limitations = tuple(
            limitation for point in report.unknowns for limitation in point.limitations
        )
        return _handoff(
            request=request,
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
            warnings=tuple(
                dict.fromkeys(
                    (
                        *report.warnings,
                        *(
                            warning
                            for handoff in specialist_handoffs
                            for warning in handoff.warnings
                        ),
                    )
                )
            ),
            limitations=tuple(dict.fromkeys((*report.limitations, *unknown_limitations))),
            confidence=(
                HandoffConfidence.MEDIUM if specialist_handoffs else HandoffConfidence.UNKNOWN
            ),
        )

    def _load_run(self, run_id: str) -> OrchestratedResearchRun:
        getter = getattr(self.run_store, "get", None)
        run = getter(run_id) if callable(getter) else None
        if not isinstance(run, OrchestratedResearchRun):
            raise ValueError("orchestrator run is unavailable")
        return run


def _handoff(
    *,
    request: DelegationRequest,
    status: OrchestratorHandoffStatus,
    started_at: datetime,
    completed_at: datetime,
    input_summary: Mapping[str, str] | None = None,
    output: Mapping[str, object] | None = None,
    evidence_ids: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    confidence: HandoffConfidence = HandoffConfidence.UNKNOWN,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff_{request.expected_kind.value}_{uuid4().hex}",
        step_id=request.step_id,
        kind=request.expected_kind,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        input_summary=dict(input_summary or {}),
        output=dict(output or {}),
        evidence_ids=evidence_ids,
        warnings=warnings,
        limitations=limitations,
        confidence=confidence,
        error_code=error_code,
        error_message=error_message,
    )


def _failed_handoff(
    request: DelegationRequest,
    started_at: datetime,
    now: datetime,
    error: Exception,
) -> AgentHandoff:
    del error
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.FAILED,
        started_at=started_at,
        completed_at=_aware_now(now),
        error_code=f"{request.role.value.replace('-', '_')}_failed",
        error_message="Specialist analysis failed safely.",
        limitations=(f"{request.role.value} specialist failed safely.",),
    )


def _provider_failure(
    request: DelegationRequest,
    started_at: datetime,
    now: datetime,
    error_code: str,
    error_message: str,
) -> AgentHandoff:
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.FAILED,
        started_at=started_at,
        completed_at=_aware_now(now),
        error_code=error_code,
        error_message=error_message,
        limitations=(error_message,),
    )


def _skipped_handoff(
    request: DelegationRequest,
    started_at: datetime,
    reason: str,
) -> AgentHandoff:
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.SKIPPED,
        started_at=started_at,
        completed_at=started_at,
        limitations=(reason,),
    )


async def _ingest_filings(
    provider: FilingProvider,
    company: FilingCompany,
    *,
    forms: tuple[str, ...],
    limit: int,
    form_limits: Mapping[str, int],
) -> tuple[FilingIngestionResult, tuple[str, ...]]:
    if not form_limits:
        return await provider.ingest_latest(company, forms=forms, limit=limit), ()
    results: list[FilingIngestionResult] = []
    errors: list[str] = []
    last_error: FilingError | None = None
    for form, form_limit in form_limits.items():
        try:
            results.append(await provider.ingest_latest(company, forms=(form,), limit=form_limit))
        except FilingError as exc:
            last_error = exc
            errors.append(f"{form} ingestion failed: {exc.message}")
    if not results:
        assert last_error is not None
        raise last_error
    return (
        FilingIngestionResult(
            company=company,
            filings=tuple(filing for result in results for filing in result.filings),
            chunks=tuple(chunk for result in results for chunk in result.chunks),
            source=results[0].source,
            warnings=tuple(
                dict.fromkeys(warning for result in results for warning in result.warnings)
            ),
        ),
        tuple(errors),
    )


def _analysis_status(value: object) -> OrchestratorHandoffStatus:
    if value in {
        FinancialReportAnalysisStatus.COMPLETE,
        StockPriceAnalysisStatus.COMPLETE,
        ContextAnalysisStatus.COMPLETE,
    }:
        return OrchestratorHandoffStatus.SUCCEEDED
    if value in {
        FinancialReportAnalysisStatus.PARTIAL,
        StockPriceAnalysisStatus.PARTIAL,
        ContextAnalysisStatus.PARTIAL,
    }:
        return OrchestratorHandoffStatus.PARTIAL
    return OrchestratorHandoffStatus.SKIPPED


def _confidence(value: object) -> HandoffConfidence:
    if value in {
        FinancialReportAnalysisStatus.COMPLETE,
        StockPriceAnalysisStatus.COMPLETE,
        ContextAnalysisStatus.COMPLETE,
    }:
        return HandoffConfidence.HIGH
    if value in {
        FinancialReportAnalysisStatus.PARTIAL,
        StockPriceAnalysisStatus.PARTIAL,
        ContextAnalysisStatus.PARTIAL,
    }:
        return HandoffConfidence.MEDIUM
    return HandoffConfidence.UNKNOWN


def _context_source_from_dict(payload: Mapping[str, object]) -> ContextSourceItem:
    return ContextSourceItem(
        id=_text(payload, "id"),
        title=_text(payload, "title"),
        summary=_text(payload, "summary"),
        source_url=_text(payload, "source_url"),
        source_name=_text(payload, "source_name"),
        source_type=ContextSourceType(_text(payload, "source_type")),
        reliability=SourceReliability(_text(payload, "reliability")),
        scope=ContextScope(_text(payload, "scope")),
        retrieved_at=_datetime(payload, "retrieved_at"),
        published_at=_optional_datetime(payload.get("published_at")),
        company_symbols=tuple(str(item) for item in _list(payload.get("company_symbols", ()))),
        sector=_optional_text(payload.get("sector")),
        region=_optional_text(payload.get("region")),
        topics=tuple(str(item) for item in _list(payload.get("topics", ()))),
        metadata={
            str(key): str(value) for key, value in _mapping(payload.get("metadata", {})).items()
        },
    )


def _text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text value must be a string")
    return value.strip() or None


def _list(value: object) -> list[object]:
    if not isinstance(value, list | tuple):
        raise ValueError("value must be a list")
    return list(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return value


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _list(value))


def _positive_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int_mapping(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    result: dict[str, int] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("mapping keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError("mapping values must be positive integers")
        result[key.strip()] = item
    return result


def _text_tuple(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in _list(payload.get(name)))
    if not values or any(not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return values


def _required_dependency(value, name: str):
    if value is None:
        raise ValueError(f"{name} is unavailable")
    return value


def _datetime(payload: Mapping[str, object], name: str) -> datetime:
    return _optional_datetime(_text(payload, name))  # type: ignore[return-value]


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _aware_now(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
